import torch
import torch.nn
from typing import Dict, Tuple
from ..model_interface import ModelInterface
from ..encoder_decoder import EncoderDecoderResult
from models.transformer_enc_dec import TransformerResult
from models.encoder_decoder import add_eos
import layers


class TransformerLMInterface(ModelInterface):
    def __init__(self, model: torch.nn.Module, label_smoothing: float = 0.0, has_token_labels: bool = False):
        self.model = model
        self.label_smoothing = label_smoothing
        self.has_token_labels = has_token_labels

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
        if normalize:
            return l.sum() / mask.sum()
        else:
            return l.sum()

    def decode_outputs(
        self, outputs: EncoderDecoderResult
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return outputs.outputs, outputs.out_lengths

    def __call__(
        self, data: Dict[str, torch.Tensor], normalize=True
    ) -> EncoderDecoderResult:
        in_len = data["in_len"].long() + 1

        sos_tensor = torch.ones((1, data["in"].shape[1])) * self.model.encoder_sos
        sos_tensor = sos_tensor.to(data["in"].device)
        inp_data = torch.cat(
            [sos_tensor, data["in"]],
            dim=0,
        ).transpose(0, 1)

        labels_key = "in" if not self.has_token_labels else "labels"

        out_data = add_eos(
            data[labels_key], data["in_len"], self.model.encoder_eos
        ).transpose(0, 1)

        # inp_data =  bs x seq_len: [SOS] a b c
        # out_data =  bs x seq_len e.g.  a b c [EOS]
        res = self.model(inp_data, in_len)

        res.data = res.data.transpose(0, 1)
        len_mask = ~self.model.generate_len_mask(inp_data.shape[1], in_len).transpose(
            0, 1
        )

        loss = self.loss(res, out_data.transpose(0, 1), len_mask, normalize)
        return EncoderDecoderResult(res.data, res.length, loss)
