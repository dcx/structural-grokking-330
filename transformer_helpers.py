import torch.nn
from layers.transformer import Transformer
from layers.transformer.transformer import TransformerDecoderWithLayer
from models import TransformerEncDecModel, TransformerDecModel
from interfaces import (
    TransformerEncDecInterface,
    TransformerDecOnlyInterface,
    TransformerLMInterface,
    TransformerHFInterface
)
from models.transformer_lm import TransformerLM
import torch.nn.functional as F


def create_lm(in_vocab_size, vec_dim, n_heads, encoder_n_layers, embedding_dropout, output_dropout, relative, rotary, causal_only, activation) -> torch.nn.Module:
    args = dict(embedding_init="xavier", scale_mode="opennmt")
    return TransformerLM(
        in_vocab_size, vec_dim, n_heads=n_heads, num_encoder_layers=encoder_n_layers, \
            embedding_dropout=embedding_dropout, output_dropout=output_dropout, tied_embedding=True, relative=relative, rotary=rotary, causal_only=causal_only,
            activation = activation, **args
    )


def create_model(
    in_vocab_size,
    out_vocab_size,
    vec_dim,
    n_heads,
    encoder_n_layers,
    decoder_n_layers,
    is_null_encoder=False,
    mode="enc_dec",
    relative=False,
    activation=F.relu
) -> torch.nn.Module:
    args = dict(embedding_init="xavier", scale_mode="opennmt", mode=mode, relative=relative)
    if is_null_encoder:
        return TransformerDecModel(
            in_vocab_size,
            out_vocab_size,
            vec_dim,
            n_heads,
            num_encoder_layers=encoder_n_layers,
            num_decoder_layers=decoder_n_layers,
            tied_embedding=True,
            **args
        )
    else:
        return TransformerEncDecModel(
            in_vocab_size,
            out_vocab_size,
            vec_dim,
            n_heads,
            num_encoder_layers=encoder_n_layers,
            num_decoder_layers=decoder_n_layers,
            tied_embedding=True,
            **args
        )


def create_model_interface(
    model, label_smoothing=0.0, is_null_encoder=False, is_lm=False, has_token_labels=False, hf=False, in_vocab=None
):
    if is_null_encoder:
        return TransformerDecOnlyInterface(model, label_smoothing=label_smoothing)
    elif hf:
        return TransformerHFInterface(model, label_smoothing=label_smoothing, encoder_sos=in_vocab(in_vocab.bos_token)["input_ids"][1],
                                      encoder_eos=in_vocab(in_vocab.eos_token)["input_ids"][1])
    elif is_lm:
        return TransformerLMInterface(model, label_smoothing=label_smoothing, has_token_labels=has_token_labels)
    else:
        return TransformerEncDecInterface(model, label_smoothing=label_smoothing)


#### Similar interfaces for pretrained models...
