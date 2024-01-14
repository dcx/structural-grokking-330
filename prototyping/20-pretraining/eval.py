import torch, torch.nn as nn, torch.utils.data as data, torch.nn.functional as F
import lightning as L
import os
import pandas as pd
from pytorch_lightning.loggers import WandbLogger

import model, dataset

# wandb setup
wandb_logger = WandbLogger(log_model="all")

# GPU setup
torch.set_float32_matmul_precision('medium')


# hyperparameters
hparams = {
    'bs': 32,
    'pad_token_id': 30, # TODO: Attach to dataset.py
    'sep_token_id': 31, # TODO: Attach to dataset.py

    # dataset
    'csv_file': '/dev/shm/amlif-50k.csv', # '../data/amlif-50k.csv',
    'use_cur_action': False,
    'use_cur_action_result': False,
    'use_next_action': False,
    'val_check_interval': 2500, # in steps
    'holdout_trees_frac': 0.15,
    'train_frac': 0.99,
    'val_frac': 0.01, # currently ignored
    'test_max_items': 2500,     
    'val_max_items': 2500,
}
# reminder: unlike main framework, here we are plugging test into val
# because Lightning (correctly) doesn't have a test step during training



hparams['model_hparams'] = {
    'd_model': 384,
    'n_enc_heads': 8,
    'n_enc_layers': 10, 
    'dropout': 0.1,
    'n_unique_tokens': 32,
    'n_output_tokens': 32,
    'lr': 1e-4,
    'pad_token_id': hparams['pad_token_id'],
    'weight_decay': 0,
    'max_steps': 10, # upper bound on BPTT steps when randomly sampling
}


# log all hyperparameters
wandb_logger.log_hyperparams(hparams)


# setup data
ds_train, ds_val, ds_test = dataset.make_datasets(
    hparams['csv_file'],
    holdout_trees_frac=hparams['holdout_trees_frac'],
    train_frac=hparams['train_frac'], val_frac=hparams['val_frac'],
    test_max_items=hparams['test_max_items'],
    val_max_items=hparams['val_max_items'],
    use_cur_action=hparams['use_cur_action'], 
    use_cur_action_result=hparams['use_cur_action_result'], 
    use_next_action=hparams['use_next_action'])

dl_train = data.DataLoader(ds_train, batch_size=hparams['bs'], collate_fn=dataset.collate_fn, shuffle=True, pin_memory=True)
dl_val = data.DataLoader(ds_val, batch_size=hparams['bs'], collate_fn=dataset.collate_fn, shuffle=True, pin_memory=True)

model = model.BPTTTransformer.load_from_checkpoint("../checkpoints/model-epoch=00-step=00055000.ckpt", **hparams['model_hparams'])
model.eval()

# predict with the model
print('test')


def detokenize(x, map_dict):
    "x: (batch, seq_len)"
    x = x.cpu().numpy()
    x = x.tolist()
    x = [[map_dict[c] for c in row] for row in x]
    x = [''.join(row) for row in x]
    return x

for i, (o, x, y, mask) in enumerate(dl_val):
    print(i)
    o = o.cuda()
    x = x.cuda()
    y = y.cuda()
    mask = mask.cuda()

    # forward pass
    n_rows = 4
    y_hat = model(o[:n_rows], x[:n_rows], mask[:n_rows])
    y_pred = y_hat.argmax(dim=-1).T # (batch, seq_len)
    y_real = y[:n_rows] # (batch, seq_len)

    # detokenize
    ins   = detokenize(x[:n_rows], ds_train.char_unmap)
    preds = detokenize(y_pred, ds_train.char_unmap)
    reals = detokenize(y_real, ds_train.char_unmap)

    # print comparison
    for inn, pred, real in zip(ins, preds, reals):
        print(inn)
        print(pred)
        print(real)
        print()

    print("Break here.")

    

