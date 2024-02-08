import torch, torch.nn as nn, torch.utils.data as data, torch.nn.functional as F
import lightning as L
import math
import torchmetrics
import random
import modelbasics as mb
import pl_bolts
import model_xf


class _TransformerDecoderLayerSwiGLU(nn.TransformerDecoderLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        norm_first: bool = False,
        gate_multiple_of: int = 128,
        device=None,
        dtype=None,
        **_kwargs,
    ) -> None:
        super().__init__(
            d_model,
            nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=F.silu,
            norm_first=norm_first,
            device=device,
            dtype=dtype,    
        )
        # Reference:
        #     https://github.com/facebookresearch/llama/blob/main/llama/model.py
        dim_feedforward = int(2 * dim_feedforward / 3)
        dim_feedforward = gate_multiple_of * (
            (dim_feedforward + gate_multiple_of - 1) // gate_multiple_of
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.linear3 = nn.Linear(d_model, dim_feedforward)

    def _ff_block(self, x):
        x = self.linear2(
            self.dropout(self.activation(self.linear1(x)) * self.linear3(x))
        )
        return self.dropout2(x)

class _TransformerEncoderLayerSwiGLU(nn.TransformerEncoderLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        norm_first: bool = False,
        gate_multiple_of: int = 128,
        **_kwargs,
    ) -> None:
        super().__init__(
            d_model,
            nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=F.silu,
            norm_first=norm_first,
        )
        # Reference:
        #     https://github.com/facebookresearch/llama/blob/main/llama/model.py
        dim_feedforward = int(2 * dim_feedforward / 3)
        dim_feedforward = gate_multiple_of * (
            (dim_feedforward + gate_multiple_of - 1) // gate_multiple_of
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.linear3 = nn.Linear(d_model, dim_feedforward)

    def _ff_block(self, x):
        x = self.linear2(
            self.dropout(self.activation(self.linear1(x)) * self.linear3(x))
        )
        return self.dropout2(x)


class S1Model(nn.Module):
    def __init__(self, d_model, n_enc_heads, n_enc_layers, n_tokens, weight_decay, pad_token_id, dropout):
        super().__init__()

        self.d_model = d_model
        self.n_enc_heads = n_enc_heads
        self.n_enc_layers = n_enc_layers
        self.n_tokens = n_tokens
        self.weight_decay = weight_decay
        self.pad_token_id = pad_token_id
        self.dropout = dropout

        self.pred_enc = nn.TransformerEncoder(mb.TransformerEncoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        self.linear = nn.Linear(d_model, n_tokens)

        # setup hooks to get activations from each layer of s1_model
        self.hooks = []
        self.activations = [None] * n_enc_layers
        def get_hook(i):
            def hook(module, input, output):
                self.activations[i] = output.reshape(1, -1, d_model) # (1, bs*seq_len, d_model)
            return hook
        for i in range(len(self.pred_enc.layers)):
            self.hooks.append(self.pred_enc.layers[i].register_forward_hook(get_hook(i)))


        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, x_enc: torch.Tensor, padding_mask: torch.Tensor=None) -> torch.Tensor:
        """
        input:
        - x_enc: (seq_len, bs, d_model)
        - padding_mask: (bs, seq_len)
        returns 
        - y_pred_enc: (seq_len, bs, d_model)
        - y_pred: (seq_len, bs, n_tokens)
        """

        # make prediction
        y_pred_enc = self.pred_enc(x_enc, src_key_padding_mask=padding_mask) # (seq_len, bs, d_model)
        y_pred = self.linear(y_pred_enc) # (seq_len, bs, n_tokens)

        return y_pred_enc, y_pred






class S2Model(nn.Module):
    """
    This module is for System 2

    Simple start: Accepts encoded game states from VAE+S1, and predicts the next action.
    """

    def __init__(self, n_bptt, d_model, n_enc_heads, n_enc_layers, n_tokens, weight_decay, pad_token_id, dropout):
        super().__init__()

        self.n_bptt = n_bptt

        self.d_model = d_model
        self.wd = weight_decay
        self.pad_token_id = pad_token_id
        self.dropout = dropout
        self.n_tokens = n_tokens

        #dummy_s2_starter = torch.rand_like(torch.zeros(d_model)) * math.sqrt(d_model)  # warning: this rand only returns positive
        #self.dummy_s2_starter = torch.nn.Parameter(dummy_s2_starter) # (d_model,)

        # models for translating to and from BPTT format
        self.enc_x_s2 = nn.TransformerEncoder(mb.TransformerEncoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        self.dec_s2_1 = nn.TransformerDecoder(mb.TransformerDecoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        self.dec_s2_2 = nn.TransformerDecoder(mb.TransformerDecoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        # self.linear_proj = nn.Linear(d_model, d_model)

        # self.dec_s2 = nn.TransformerDecoder(mb.TransformerDecoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        self.linear_s2 = nn.Linear(d_model, n_tokens)

        # # models for game thinking
        # # self.enc_bptt = nn.TransformerEncoder(mb.TransformerEncoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        # self.enc_step1 = nn.TransformerEncoder(mb.TransformerEncoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        # self.dec_bptt = nn.TransformerDecoder(mb.TransformerDecoderLayer(d_model=d_model, nhead=n_enc_heads*2, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)


        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear_s2.bias.data.zero_()
        self.linear_s2.weight.data.uniform_(-initrange, initrange)

        self.dummy_s2_starter.data.uniform_(-initrange, initrange)

    def forward(self, x_enc: torch.Tensor, s1_model) -> torch.Tensor:
        """
        x_enc: (1, bs*seq_len, d_model)
        s1_activations: (n_enc_layers, bs*seq_len, d_model)
        returns y_pred: (n_bptt, seq_len, bs, n_tokens)

        No padding_mask required because we're not doing any attention over the sequence.
        So cross-enteropy loss will handle ignoring the padding tokens.
        """
        _, bs_sl, d_model = x_enc.shape

        # prep output tensor
        s2_pred = torch.zeros(self.n_bptt, 1, bs_sl, self.n_tokens, device=self.device) # (n_bptt, 1, bs_sl, n_tokens)

        # prep s2 state
        cur_x_enc = x_enc # (1, bs_sl, d_model)
        s2_state = self.dummy_s2_starter.clone().repeat(1, bs_sl, 1) # (1, bs_sl, d_model)

        for i in range(self.n_bptt):
            # PHASE 1/2: make s1 query and update state
            s2_enc = self.enc_x_s2(cur_x_enc) # (1, bs_sl, d_model)
            s1_fwd = s1_model.pred_enc(cur_x_enc).detach() # (1, bs_sl, d_model)
            if i == 0:
                s1_save = s1_fwd
            s1_activations = torch.cat(s1_model.activations, dim=0).detach() # (n_layers, bs_sl, d_model)
            s2_phase_1 = self.dec_s2_1(torch.cat((s2_state, s2_enc), dim=0), s1_activations) # (2, bs_sl, d_model)
            s2_pred[i] = self.linear_s2(s1_save + s2_phase_1[-1:]) # (1, bs_sl, n_tokens)

            # PHASE 2/2: glance at original x_enc, make S1 query
            s2_update = self.dec_s2_2(s2_phase_1, cur_x_enc) # (2, bs_sl, d_model)
            s2_state = s2_update[:1] # (1, bs_sl, d_model)
            cur_x_enc = s2_update[1:] # (1, bs_sl, d_model)

        # for i in range(8):
        #     # PHASE 1/2: update state
        #     # make s1 query (frozen model)
        #     s1_query = s2_state[-1:] # (1, bs*seq_len, d_model)
        #     s1_resp = self.s1_model.pred_enc(s1_query) # (1, bs*seq_len, d_model)
        #     s1_acts = torch.cat(self.s1_model_activations, dim=0) # (n_layers, bs*seq_len, d_model)
        #     # make x decoder input (encode)
        #     s2_state_enc = torch.zeros(2, bs*seq_len, d_model, device=x_enc.device) # (2, bs*seq_len, d_model)
        #     s2_state_enc[0, :] = s2_state[0, :]
        #     s2_state_enc[1, :] = self.enc_x(s2_state[:1]) # (bs*seq_len, d_model)
        #     # push through decoder: update state
        #     s2_phase1 = self.dec_s2_1(s2_state_enc, s1_acts) # (2, bs*seq_len, d_model)
        #     # PHASE 2/2: glance at original x_enc and update state for next cycle
        #     s2_state = self.dec_s2_2(s2_phase1, x_enc) # (2, bs*seq_len, d_model)

        # # make output pred
        # y_pred = self.linear_out(s2_phase1[1:]) # (bs*seq_len, n_tokens)

        # # shape for output
        # y_pred = y_pred.reshape(seq_len, bs, -1).unsqueeze(0) # (1, seq_len, bs, n_tokens)
        # return y_pred # (n_bptt, seq_len, bs, n_tokens)
        return s2_pred



# Define entire system as a LightningModule
class S2Transformer(L.LightningModule):
    def __init__(self, vae_model, d_model, n_enc_heads, n_enc_layers, n_loop_layers, n_tokens, lr_s1, lr_s2, weight_decay, pad_token_id, predictive, dropout, n_bptt, max_bs, bptt_every_loop, n_preds, grad_acc, step_bias, d_loop_feed_forward, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.d_model = d_model
        self.d_loop_feed_forward = d_loop_feed_forward
        self.lr_s1 = lr_s1
        self.lr_s2 = lr_s2
        self.wd = weight_decay
        self.pad_token_id = pad_token_id
        self.predictive = predictive
        self.metrics = {}
        self.dropout = dropout
        self.n_bptt = n_bptt
        self.n_tokens = n_tokens
        self.bptt_every_loop = bptt_every_loop
        self.initrange = 0.1
        self.n_preds = n_preds # number of predictions to make per token
        self.grad_acc = grad_acc
        self.step = 0
        self.step_bias = step_bias

        # # freeze pre-trained vae
        # self.vae = vae_model
        # for param in self.vae.parameters():
        #     param.requires_grad = False

        self.loop_d_scale = 1
        loop_h_scale = 1
        in_h_scale = 1
        # self.d_vae = 128
        # d_vae = 128

        self.embedding = nn.Embedding(n_tokens, d_model, padding_idx=pad_token_id)
        #self.in_enc = nn.TransformerEncoder(_TransformerEncoderLayerSwiGLU(d_model=d_model, nhead=n_enc_heads, dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        self.in_enc = nn.TransformerEncoder(mb.TransformerEncoderLayer(d_model=d_model, nhead=n_enc_heads, activation=F.leaky_relu, dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)

        # self.in_linear = nn.Linear(d_vae, d_model*self.loop_d_scale)
        self.in_model = nn.TransformerEncoder(_TransformerEncoderLayerSwiGLU(d_model=d_model*self.loop_d_scale, nhead=n_enc_heads*in_h_scale, dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)

        #self.loop_starter = nn.TransformerEncoder(_TransformerEncoderLayerSwiGLU(d_model=d_model*self.loop_d_scale, nhead=n_enc_heads*loop_h_scale, dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        self.loop_starter = nn.TransformerEncoder(mb.TransformerEncoderLayer(d_model=d_model*self.loop_d_scale, nhead=n_enc_heads*loop_h_scale, activation=F.leaky_relu, dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)

        # self.loop_model_1 = nn.TransformerDecoder(_TransformerDecoderLayerSwiGLU(d_model=d_model*self.loop_d_scale, nhead=n_enc_heads*loop_h_scale, dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        # self.loop_model_2 = nn.TransformerDecoder(_TransformerDecoderLayerSwiGLU(d_model=d_model*self.loop_d_scale, nhead=n_enc_heads*loop_h_scale, dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        # self.loop_model_3 = nn.TransformerDecoder(_TransformerDecoderLayerSwiGLU(d_model=d_model*self.loop_d_scale, nhead=n_enc_heads*loop_h_scale, dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        # self.loop_model_4 = nn.TransformerDecoder(_TransformerDecoderLayerSwiGLU(d_model=d_model*self.loop_d_scale, nhead=n_enc_heads*loop_h_scale, dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        # self.loop_model = model_xf.get_decoder(d_model=d_model*self.loop_d_scale)
        #self.loop_model = nn.TransformerDecoder(_TransformerDecoderLayerSwiGLU(d_model=d_model*self.loop_d_scale, dim_feedforward=d_loop_feed_forward, nhead=n_enc_heads*loop_h_scale, dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        self.loop_model = nn.TransformerDecoder(mb.TransformerDecoderLayer(d_model=d_model*self.loop_d_scale, dim_feedforward=d_loop_feed_forward, nhead=n_enc_heads*loop_h_scale, activation=F.leaky_relu, dropout=self.dropout, batch_first=False), num_layers=n_loop_layers)

        self.pred_in_linear = nn.Linear(d_model*self.loop_d_scale, d_model)


        self.backprop_to_ingest = False # Leave as False. After testing, True starts better but eventually perf is consistently worse (0.3% absolute at the 0.77 mark)


        self.pred_linear = nn.Linear(d_model, n_tokens)
        if self.n_preds > 1:
            self.pred_tf = nn.TransformerDecoder(_TransformerDecoderLayerSwiGLU(d_model=d_model, nhead=n_enc_heads, dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
            dummy_s2_starters = (torch.rand_like(torch.zeros(self.n_preds,1,d_model)) * (self.initrange*2)) - self.initrange
            self.dummy_s2_starters = torch.nn.Parameter(dummy_s2_starters) # (n_preds,1,d_model)
        else:
            self.pred_model = nn.TransformerEncoder(_TransformerEncoderLayerSwiGLU(d_model=d_model, nhead=n_enc_heads, dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)

        dummy_wm = (torch.rand_like(torch.zeros(self.n_bptt,d_model)) * (self.initrange*2)) - self.initrange # (n_bptt,d_model)
        self.dummy_wm = torch.nn.Parameter(dummy_wm) # (n_bptt,d_model)

        # metrics
        for mode in ['train', 'val']: # hack: metrics must be on self or Lightning doesn't handle their devices correctly
            for i in range(n_bptt+int(self.backprop_to_ingest)):
                setattr(self, f'{mode}_acc_{i:02}', torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_tokens, ignore_index=pad_token_id))
                self.metrics[f'{mode}_acc_{i:02}'] = getattr(self, f'{mode}_acc_{i:02}')

        self.init_weights()

    def init_weights(self):
        initrange = self.initrange
        #self.in_linear.bias.data.zero_()
        #self.in_linear.weight.data.uniform_(-initrange, initrange)

        self.pred_in_linear.bias.data.zero_()
        self.pred_in_linear.weight.data.uniform_(-initrange, initrange)

        self.pred_linear.bias.data.zero_()
        self.pred_linear.weight.data.uniform_(-initrange, initrange)

    def model_step(self, batch, batch_idx, mode='train'):
        s1_opt = self.optimizers()
        s1_lr_sched = self.lr_schedulers()

        x, y = batch # (bs, seq_len)
        padding_mask = (x==self.pad_token_id) # (bs, seq_len)
        x, y = x.to(torch.long), y.to(torch.long) # do conversion on GPU (mem bottleneck)

        # encode to single token with pre-trained, frozen VAE
        # x_emb, x_enc = self.vae.forward_enc(x, padding_mask=padding_mask) # (seq_len, bs, d_model)

        # make causally encoded tokens
        x_emb = self.embedding(x) # (bs, seq_len, d_model)
        cmask = torch.nn.Transformer.generate_square_subsequent_mask(sz=x_emb.shape[1]).to(x.device) # (seq_len, seq_len)
        x_in = self.in_enc(x_emb.transpose(0,1), src_key_padding_mask=padding_mask, is_causal=True, mask=cmask) # (seq_len, bs, d_model)

        seq_len, bs, d_model = x_in.shape

        # since each token is its own row, convert to seq_len=1 for hygiene
        # (this makes padding irrelevant, since cross-entropy loss ignores padding tokens,
        # but it's a good habit to keep the shape consistent)
        x_in = x_in.reshape(1, -1, d_model) # (1, seq_len*bs, d_vae)
        #y = y.reshape(-1) # (bs*seq_len,)
        padding_mask = padding_mask.reshape(-1,1) # (seq_len*bs,1)

        # translate VAE inputs into internal format
        #x_in = self.in_linear(x_enc) # (1, bs*seq_len, 2*d_model)
        #x_in = self.in_model(x_in) # (1, bs*seq_len, 2*d_model)

        # prime the pump: take a first glance at the input and produce the internal representation
        # for loops to come
        x_wip = self.loop_starter(x_in) # (1, bs*seq_len, 2*d_model)

        # x_in = x_enc
        # x_wip = x_in


        # prep tensor to hold loop results
        x_loops = torch.zeros(self.n_bptt, 1, bs*seq_len, self.d_model*self.loop_d_scale, device=self.device) # (n_bptt, 1, bs*seq_len, 2*d_model)

        # # run loop model n_loops times
        # x_wip = x_wip + self.loop_model_1(x_wip, x_in) # (1, bs*seq_len, 2*d_model)
        # x_loops[0] = x_wip
        # x_wip = x_wip + self.loop_model_2(x_wip, x_in) # (1, bs*seq_len, 2*d_model)
        # x_loops[1] = x_wip
        # x_wip = x_wip + self.loop_model_3(x_wip, x_in) # (1, bs*seq_len, 2*d_model)
        # x_loops[2] = x_wip
        # x_wip = x_wip + self.loop_model_4(x_wip, x_in) # (1, bs*seq_len, 2*d_model)
        # x_loops[3] = x_wip

        for i in range(self.n_bptt):
            x_wip = self.loop_model(x_wip, x_in) # (1, bs*seq_len, 2*d_model)
            x_loops[i] = x_wip

        
        bpti = 0
        if self.backprop_to_ingest:
            x_loops = torch.cat((x_in.unsqueeze(0), x_loops), dim=0) # (n_bptt+1, 1, bs*seq_len, 2*d_model)
            bpti = 1

        # downsize to prediction format
        x_loops = self.pred_in_linear(x_loops) # (n_bptt+bpti, 1, bs*seq_len, d_model)

        # translate internal format into prediction
        x_loops = x_loops.transpose(0, 1).reshape(1, -1, d_model) # (1, (n_bptt+bpti)*bs*seq_len, d_model)

        if self.n_preds > 1:
            s1_pred_model_out = self.pred_tf(self.dummy_s2_starters.repeat(1,(self.n_bptt+bpti)*bs*seq_len,1), x_loops) # (n_preds, n_bptt*bs*seq_len, d_model)
        else:
            s1_pred_model_out = self.pred_model(x_loops) # (n_preds=1, n_bptt*bs*seq_len, d_model)

        s1_pred = self.pred_linear(s1_pred_model_out) # (n_preds, n_bptt*bs*seq_len, n_tokens)
        
        # calculate loss
        s1_pred = s1_pred.reshape(self.n_preds, self.n_bptt+bpti, seq_len, bs, -1) # (n_preds, n_bptt, seq_len, bs, n_tokens)
        s1_pred = s1_pred.permute(0, 1, 3, 2, 4) # (n_preds, n_bptt, bs, seq_len, n_tokens)
        s1_pred = s1_pred.reshape(self.n_preds, -1, seq_len, self.n_tokens) # (n_preds, n_bptt*bs, seq_len, n_tokens)

        # # trim seq_len by n_preds-1 since we need this many forward
        # s1_pred = s1_pred[:,:-self.n_preds+1] # (n_preds*n_bptt*bs, (seq_len-n_preds+1), n_tokens)
        # # pull out y for this many forward
        # y_fwd = y.unsqueeze(0).unsqueeze(0).repeat(self.n_preds, 1, 1, 1) # (n_preds, n_bptt, bs, seq_len)
        # y_fwd = y_fwd[:,:,:,-self.n_preds+1] # (n_preds, n_bptt, bs, seq_len-n_preds+1)

        # step bias:((n_bppt+bpti),) where each value is (1+step_bias)^i for step
        # e.g. if step_bias=0.1, then step_bias_multiplier = [1, 1.1, 1.21, 1.331, 1.4641, ...]
        i_range = torch.arange(self.n_bptt+bpti, device=self.device, dtype=s1_pred.dtype) # (n_bptt+bpti,)
        i_range = i_range.unsqueeze(1) # (n_bptt+bpti,1)
        i_range = i_range.repeat(1,bs) # (n_bptt+bpti,bs)
        i_range = i_range.reshape(-1) # ((n_bppt+bpti)*bs,)
        step_bias_multiplier = (1+self.step_bias)**i_range # ((n_bppt+bpti)*bs,)
        step_bias_multiplier = step_bias_multiplier.unsqueeze(1) # ((n_bppt+bpti)*bs,1)

        # quick and dirty version to test if this even works
        s1_pred_ce = s1_pred.permute(0, 1, 3, 2) # (n_preds, n_bptt*bs, n_tokens, seq_len)

        s1_pred_cur = s1_pred_ce[0] # (n_bptt*bs, n_tokens, seq_len)
        s1_loss = F.cross_entropy(s1_pred_cur, y.repeat(self.n_bptt+bpti, 1), ignore_index=self.pad_token_id, reduction='none') # (n_bptt*bs, seq_len)
        s1_loss = (s1_loss * step_bias_multiplier).mean() # (1,)

        for i in range(1,self.n_preds):
            s1_pred_cur = s1_pred_ce[i][:,:,:-i] # (n_bptt*bs, n_tokens, seq_len)
            cur_loss = F.cross_entropy(s1_pred_cur, y.repeat((self.n_bptt+bpti), 1)[:,i:], ignore_index=self.pad_token_id, reduction='none') # (n_bptt*bs, seq_len-i)
            s1_loss += (cur_loss * step_bias_multiplier).mean() # (1,)

        s1_loss /= self.n_preds
        # log stats for each loop
        # s1_pred = s1_pred.reshape(self.n_bptt*bs, seq_len, -1) # (n_bptt*bs, seq_len, n_tokens)
        _, s1_y_hat = torch.max(s1_pred, dim=3) # (n_preds, n_bptt*bs, seq_len)
        s1_y_hat = s1_y_hat.reshape(self.n_preds, self.n_bptt+bpti, bs, seq_len) # (n_preds, n_bptt, bs, seq_len)
        for i in range(self.n_bptt+bpti):
            s1_acc = self.metrics[f'{mode}_acc_{i:02}'](s1_y_hat[0,i], y)
            for j in range(1,self.n_preds): 
                s1_acc_j = self.metrics[f'{mode}_acc_{i:02}'](s1_y_hat[j,i][:,:-j], y[:,j:]) # compare the second prediction against the second y, but only for the tokens actually have a second y (i.e. not the last j tokens per sequence)
                s1_acc += s1_acc_j
            self.log(f"{mode}_acc_{i:02}", s1_acc/self.n_preds, sync_dist=True)
                # s1_acc = self.metrics[f'{mode}_acc_{i}'](s1_y_hat[i], y)
                # self.log(f"{mode}_acc_{i}", s1_acc, sync_dist=True)

        #s1_acc = self.metrics[f'{mode}_acc_{i}'](s1_y_hat, y)
        #self.log(f"{mode}_acc_{i}", s1_acc)


        #     # translate internal format into prediction
        #     s1_pred_enc = self.pred_model(x_wip) # (1, bs*seq_len, d_model)
        #     s1_pred = self.pred_linear(s1_pred_enc) # (1, bs*seq_len, n_tokens)

        #     # calculate loss
        #     s1_pred = s1_pred.reshape(seq_len, bs, -1) # (seq_len, bs, n_tokens)
        #     s1_pred = s1_pred.permute(1, 0, 2) # (bs, seq_len, n_tokens)
        #     s1_pred = s1_pred.reshape(-1, seq_len, self.n_tokens) # (bs, seq_len, n_tokens)

        #     s1_pred_ce = torch.permute(s1_pred, (0, 2, 1)) # (bs, n_tokens, seq_len)
        #     s1_loss += F.cross_entropy(s1_pred_ce, y, ignore_index=self.pad_token_id)

        #     # log stats for each loop
        #     s1_pred = s1_pred.reshape(bs, seq_len, -1) # (bs, seq_len, n_tokens)
        #     _, s1_y_hat = torch.max(s1_pred, dim=2) # (bs, seq_len)
        #     s1_acc = self.metrics[f'{mode}_acc_{i}'](s1_y_hat, y)
        #     self.log(f"{mode}_acc_{i}", s1_acc)

        # s1_loss /= self.n_bptt

        # optimize
        if mode == 'train':
            s1_opt.zero_grad()
            self.manual_backward(s1_loss)
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            if self.step % self.grad_acc == 0:
                s1_opt.step()
                s1_lr_sched.step()

            self.step = (self.step + 1) % self.grad_acc

        self.log_dict({"s1_loss": s1_loss}, prog_bar=True, sync_dist=True)


        # # ===========================
        # # Pull out low-confidence predictions
        # # ===========================

        # s2_fraction = 0.25
        # n_lc = int(bs*seq_len*s2_fraction)
        # s1_prob = F.softmax(s1_pred_enc, dim=2) # (1, bs*seq_len, n_tokens)
        # s1_maxprob, _ = torch.max(s1_prob, dim=2) # (1, bs*seq_len)
        # s1_maxprob_nonpad = s1_maxprob * ~padding_mask.squeeze(1) # (1, bs*seq_len)

        # s1_max_idxs = torch.argsort(s1_maxprob_nonpad, descending=True, dim=1)[:,:n_lc] # (n_lc,)
        # lc_x_enc = x_enc[:,s1_max_idxs[0]] # (1, n_lc, d_model)
        # lc_y = y.T.reshape(-1)[s1_max_idxs[0]] # (n_lc,)


        # # ===========================
        # # Predict, optimize S2 model
        # # ===========================

        # s2_lc_pred = self.s2_model(lc_x_enc, self.s1_model) # (n_bptt, 1, n_lc, n_tokens)

        # if not self.bptt_every_loop:
        #     s2_lc_pred = s2_lc_pred[-1:] # (1, 1, n_lc, n_tokens)
        #     s2_lc_pred_ce = s2_lc_pred.squeeze(0).squeeze(0) # (n_lc, n_tokens)
        # else:
        #     s2_lc_pred = s2_lc_pred.squeeze(1)
        #     s2_lc_pred_ce = s2_lc_pred.reshape(-1, self.n_tokens) # (n_bptt*n_lc, n_tokens)
        #     lc_y = lc_y.repeat(self.n_bptt) # (n_bptt*n_lc,)

        # s2_loss = F.cross_entropy(s2_lc_pred_ce, lc_y, ignore_index=self.pad_token_id)

        # # log stats
        # _, s2_lc_y_hat = torch.max(s2_lc_pred_ce, dim=1) # (n_bptt*n_lc,)
        # s2_lc_acc = self.metrics[f'{mode}_s2_lc_acc'](s2_lc_y_hat, lc_y)
        # self.log(f"{mode}_s2_lc_acc", s2_lc_acc)

        # # optimize
        # if mode == 'train':
        #     s2_opt.zero_grad()
        #     self.manual_backward(s2_loss)
        #     torch.nn.utils.clip_grad_norm_(self.s2_model.parameters(), 1.0)
        #     s2_opt.step()
        #     s2_lr_sched.step()

        # # recombine with S1 preds to report overall accuracy
        # cmb_y_hat = s1_y_hat.T.reshape(-1) # (bs*seq_len,)
        # cmb_y_hat[s1_max_idxs[0]] = s2_lc_y_hat[-n_lc:] # (bs*seq_len,): use last bptt period only
        # cmb_acc = self.metrics[f'{mode}_cmb_acc'](cmb_y_hat, y.T.reshape(-1))
        # self.log(f"{mode}_cmb_acc", cmb_acc)
        
        # self.log(f"{mode}_loss_delta", s1_loss - s2_loss)
        # self.log(f"{mode}_acc_delta", cmb_acc - s1_acc)

        # self.log_dict({"s1_loss": s1_loss, "s2_loss": s2_loss}, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.model_step(batch, batch_idx, mode='train')
    
    def validation_step(self, batch, batch_idx):
        return self.model_step(batch, batch_idx, mode='val')

    def log_and_reset_metrics(self, mode):
        for metric_name, metric in self.metrics.items():
            if metric_name.startswith(mode):
                self.log(f"{metric_name}_epoch", metric.compute())
                metric.reset()

    def on_train_epoch_end(self):
        self.log_and_reset_metrics('train')

    def on_val_epoch_end(self):
        self.log_and_reset_metrics('val')

    def configure_optimizers(self):
        s1_opt = torch.optim.AdamW(self.parameters(), lr=self.lr_s1, weight_decay=self.wd)
        # s2_opt = torch.optim.AdamW(self.s2_model.parameters(), lr=self.lr_s2, weight_decay=self.wd)
        s1_lr_sched = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(
            s1_opt, warmup_epochs=2000, max_epochs=100000, warmup_start_lr=0.0, eta_min=0.1*self.lr_s1, last_epoch=-1)
        # s2_lr_sched = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(
        #     s2_opt, warmup_epochs=2000, max_epochs=100000, warmup_start_lr=0.0, eta_min=0.1*self.lr_s2, last_epoch=-1)

        return [s1_opt], [s1_lr_sched] # [s1_opt, s2_opt], [s1_lr_sched, s2_lr_sched] 

