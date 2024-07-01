import torch
import torch.nn
import torch.nn.functional as F
from typing import Optional
from .multi_head_attention import AttentionMask, MultiHeadAttentionBase
import layers
import math
from rotary_embedding_torch import RotaryEmbedding
import pdb

class RotaryMultiheadAttention(MultiHeadAttentionBase):
    def __init__(self, state_size: int, n_heads: int, dropout: float = 0.0, global_pos_bias: bool = True,
                 global_content_bias: bool = True, input_size: Optional[int] = None, causal_only: bool = False):
        super().__init__(state_size, n_heads, dropout)

        self.data_to_kv = torch.nn.Linear(state_size, 2 * n_heads * self.projection_size, bias=False)
        self.data_to_q = torch.nn.Linear(state_size if input_size is None else input_size,
                                         n_heads * self.projection_size, bias=False)

        self.global_content_bias = torch.nn.Parameter(torch.zeros([n_heads, self.projection_size])) \
                                   if global_content_bias else None
        self.global_pos_bias = torch.nn.Parameter(torch.zeros([n_heads, self.projection_size])) \
                               if global_pos_bias else None
        self.dropout_p = dropout

        self.pos_to_pq = RotaryEmbedding(dim = state_size // n_heads)
        self.causal_only = causal_only

    def add_head_specific_bias(self, data: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        # data [batch * n_heads, len, c]
        # bias [n_heads, c]
        return (data.view(-1, bias.shape[0], *data.shape[1:]) + bias.unsqueeze(1).type_as(data)).view_as(data) \
               if bias is not None else data
    
    def get_attn_mask(self, mask):
        # return True when attended to, False otherwise
        return ~mask.position_mask.unsqueeze(0) * ~mask.src_length_mask.unsqueeze(1)

    def forward(self, curr_state: torch.Tensor, attend_to: torch.Tensor, mask: Optional[AttentionMask],
                pos_offset: int = 0, need_weights: bool = False):
        # curr_state: [batch_size, out_len, c]
        # attend_to: [batch_size, in_len, c]
        batch_size, in_len = attend_to.shape[0:2]
        out_len = curr_state.shape[1]

        k_content, v = self.transform_data(attend_to, self.data_to_kv, 2)
        q, = self.transform_data(curr_state, self.data_to_q, 1)
        q_content = self.add_head_specific_bias(q, self.global_content_bias)

        # check dims here
        q_rot = torch.reshape(q_content, (batch_size, self.n_heads, out_len, -1))
        k_rot = torch.reshape(k_content, (batch_size, self.n_heads, out_len, -1))

        q_pos = self.pos_to_pq.rotate_queries_or_keys(q_rot)
        k_pos = self.pos_to_pq.rotate_queries_or_keys(k_rot)

        q_pos = torch.reshape(q_pos, (batch_size, self.n_heads, out_len, -1))
        k_pos = torch.reshape(k_pos, (batch_size, self.n_heads, out_len, -1))

        # Use sdpa now
        v = torch.reshape(v, (batch_size, self.n_heads, out_len, -1))

        if self.causal_only:
            # use the fast kernel with causal-only mask
            # mask out padded tokens afterwards

            data = F.scaled_dot_product_attention(q_pos, k_pos, v, is_causal = True, dropout_p = self.dropout_p)

        else:
            attn_mask = self.get_attn_mask(mask).unsqueeze(1).repeat(1, self.n_heads, 1, 1)
            
            data = F.scaled_dot_product_attention(q_pos, k_pos, v, attn_mask, dropout_p = self.dropout_p)

        data = (
            data.view(batch_size, self.n_heads, out_len, -1)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, out_len, -1)
        )

        if self.causal_only:
            # mask out [PAD] representations
            logits = logits.masked_fill(
                mask.src_length_mask.unsqueeze(-1), float("-inf")
            )

        return data

    def reset_parameters(self):
        super().reset_parameters()

        torch.nn.init.xavier_uniform_(self.data_to_q.weight)
        torch.nn.init.xavier_uniform_(self.pos_to_pq.weight)
        torch.nn.init.xavier_uniform_(self.data_to_kv.weight[:self.data_to_kv.weight.shape[0]//2])
        torch.nn.init.xavier_uniform_(self.data_to_kv.weight[self.data_to_kv.weight.shape[0]//2:])

        if self.global_content_bias is not None:
            self.global_content_bias.fill_(0)

        if self.global_pos_bias is not None:
            self.global_pos_bias.fill_(0)
