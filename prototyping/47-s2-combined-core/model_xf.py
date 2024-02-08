
from xformers.factory.model_factory import xFormer, xFormerConfig
import torch

SEQ = 16
BATCH = 16


def get_decoder(d_model=768, d_feed_forward=2048):
    # https://github.com/facebookresearch/xformers/blob/main/HOWTO.md#pytorch-transformer

    my_config = [
        # A list of the encoder or decoder blocks which constitute the Transformer.
        # Note that a sequence of different encoder blocks can be used, same for decoders
        {
            "reversible": False,  # Optionally make these layers reversible, to save memory
            "block_type": "encoder",
            "num_layers": 1,  # Optional, this means that this config will repeat N times
            "dim_model": d_model,
            "residual_norm_style": "post",  # Optional, pre/post
            "multi_head_config": {
                "num_heads": 8,
                "residual_dropout": 0,
                "attention": {
                    "name": "scaled_dot_product",  # whatever attention mechanism
                    "dropout": 0,
                    "causal": False,
                    "seq_len": SEQ,
                },
            },
            "feedforward_config": {
                "name": "MLP",
                "dropout": 0,
                "activation": "relu",
                "hidden_layer_multiplier": 4,
            },
        },
        {
            "reversible": False,  # Optionally make these layers reversible, to save memory
            "block_type": "decoder",
            "num_layers": 8,  # Optional, this means that this config will repeat N times
            "dim_model": d_model,
            "residual_norm_style": "post",  # Optional, pre/post
            "multi_head_config_masked": {
                "num_heads": 8,
                "residual_dropout": 0,
                "attention": {
                    "name": "scaled_dot_product",  # whatever attention mechanism
                    "dropout": 0,
                    "causal": False,
                    "seq_len": SEQ,
                },
            },
            "multi_head_config_cross": {
                "num_heads": 8,
                "residual_dropout": 0,
                "attention": {
                    "name": "scaled_dot_product",  # whatever attention mechanism
                    "dropout": 0,
                    "causal": False,
                    "seq_len": SEQ,
                },
            },
            "feedforward_config": {
                "name": "MLP",
                "dropout": 0,
                "activation": "relu",
                "hidden_layer_multiplier": 4,
            },
        },
    ]

    # This part of xFormers is entirely type checked and needs a config object,
    # could be changed in the future
    config = xFormerConfig(my_config)
    model = xFormer.from_config(config)

    return model


if __name__ == "__main__":
    d_model = 512
    model = get_decoder(d_model=d_model)
    print(model)

    #  Test out with dummy inputs
    x = torch.rand((BATCH, SEQ, d_model)) # .abs().to(torch.int)
    print(x.shape)
    y = model(src=x, tgt=x)
    print(y)