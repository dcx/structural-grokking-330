import torch
import torch.nn
from typing import Dict, Tuple
from ..model_interface import ModelInterface
from ..encoder_decoder import EncoderDecoderResult
from models.transformer_enc_dec import TransformerResult
from models.encoder_decoder import add_eos, add_eos_pack
import layers
import pdb


class TransformerLMInterface(ModelInterface):
    def __init__(self, model: torch.nn.Module, label_smoothing: float = 0.0, has_token_labels: bool = False):
        self.model = model
        self.label_smoothing = label_smoothing
        self.has_token_labels = has_token_labels
        self.encoder_sos = self.model.encoder_sos
        self.encoder_eos = self.model.encoder_eos

    def loss(
        self,
        outputs: TransformerResult,
        ref: torch.Tensor,
        mask: torch.Tensor,
        normalize,
    ) -> torch.Tensor:
        l = layers.cross_entropy( 
            # TODO: Set the ignore_index in a cleaner way 
            # (e.g. linked to Vocabulary settings)
            outputs.data, ref, reduction="none", smoothing=self.label_smoothing, ignore_index=0
        )
        l = l.reshape_as(ref) * mask
        # pdb.set_trace()
        if normalize:
            return l.sum() / mask.sum()
        else:
            return l.sum()

    def decode_outputs(
        self, outputs: EncoderDecoderResult
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return outputs.outputs, outputs.out_lengths

    def __call__(
        self, data: Dict[str, torch.Tensor], normalize=True, pack=False
    ) -> EncoderDecoderResult:
        in_len = data["in_len"].long() + (1 - int(pack)) # in_len for packing included SOS

        # will skip this with packing
        if not pack:
            sos_tensor = torch.ones((1, data["in"].shape[1])) * self.model.encoder_sos
            sos_tensor = sos_tensor.to(data["in"].device)
            inp_data = torch.cat(
                [sos_tensor, data["in"]],
                dim=0,
            ).transpose(0, 1)
        else:
            # Handled by the PackingDataset
            inp_data = data["in"].transpose(0,1)
        # print(inp_data)

        labels_key = "in" if not self.has_token_labels else "labels"

        if not pack:
            out_data = add_eos(
                data[labels_key], data["in_len"], self.model.encoder_eos
            ).transpose(0, 1)
        else:
            out_data = add_eos_pack(
                data[labels_key], data["in_len"], self.model.encoder_eos, self.model.encoder_sos
            ).transpose(0, 1)
        # print(out_data)

        # recreate targets with packing

        # inp_data =  bs x seq_len: [SOS] a b c
        # out_data =  bs x seq_len e.g.  a b c [EOS]
        res = self.model(inp_data, in_len)

        res.data = res.data.transpose(0, 1)
        len_mask = ~self.model.generate_len_mask(inp_data.shape[1], in_len).transpose(
            0, 1
        )

        loss = self.loss(res, out_data.transpose(0, 1), len_mask, normalize)
        return EncoderDecoderResult(res.data, res.length, loss)

class TransformerHFInterface(ModelInterface):
    def __init__(self, model: torch.nn.Module, label_smoothing: float = 0.0, encoder_sos=-1, encoder_eos=-1):
        self.model = model
        self.label_smoothing = label_smoothing
        self.encoder_sos = encoder_sos
        self.encoder_eos = encoder_eos

    def loss(
        self,
        outputs: TransformerResult,
        ref: torch.Tensor,
        mask: torch.Tensor,
        normalize,
    ) -> torch.Tensor:
        l = layers.cross_entropy( 
            # TODO: Set the ignore_index in a cleaner way 
            # (e.g. linked to Vocabulary settings)
            outputs, ref, reduction="none", smoothing=self.label_smoothing, ignore_index=0
        )
        l = l.reshape_as(ref) * mask
        # pdb.set_trace()
        if normalize:
            return l.sum() / mask.sum()
        else:
            return l.sum()

    def generate_mask(self, max_len, in_lens):
        return torch.arange(max_len).expand(len(in_lens), max_len).to(in_lens.device) < in_lens.unsqueeze(1)

    def decode_outputs(
        self, outputs: EncoderDecoderResult
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return outputs.outputs, outputs.out_lengths

    def __call__(
        self, data: Dict[str, torch.Tensor], normalize=True, pack=False
    ) -> EncoderDecoderResult:
        in_len = data["in_len"].long() + (1 - int(pack)) # in_len for packing included SOS

        # will skip this with packing
        if not pack:
            sos_tensor = torch.ones((1, data["in"].shape[1]), dtype=torch.long) * self.encoder_sos
            sos_tensor = sos_tensor.to(data["in"].device)
            inp_data = torch.cat(
                [sos_tensor, data["in"]],
                dim=0,
            ).transpose(0, 1)
        else:
            inp_data = data["in"].transpose(0,1)
        # print(inp_data)

        labels_key = "in"

        if not pack:
            out_data = add_eos(
                data[labels_key], data["in_len"], self.encoder_eos
            ).transpose(0, 1)
        else:
            out_data = add_eos_pack(
                data[labels_key], data["in_len"], self.encoder_eos, self.encoder_sos
            ).transpose(0, 1)
        # print(out_data)

        # recreate targets with packing

        # inp_data =  bs x seq_len: [SOS] a b c
        # out_data =  bs x seq_len e.g.  a b c [EOS]
        attn_mask = self.generate_mask(inp_data.shape[1], in_len).to(inp_data.device)
        out = self.model(inp_data, attention_mask=attn_mask)

        res = out["logits"]
        res = res.transpose(0, 1)
        len_mask = attn_mask.transpose(
            0, 1
        )

        loss = self.loss(res, out_data.transpose(0, 1), len_mask, normalize)
        return EncoderDecoderResult(res, in_len, loss)