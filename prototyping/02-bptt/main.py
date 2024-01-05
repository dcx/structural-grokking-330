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
    'pad_token_id': 2,

    # dataset
    'val_check_interval': 800, # in steps
    'n_train': 50000,     
    'n_val': 800,     

    'n_automata_steps': 4,
}
# reminder: unlike main framework, here we are plugging test into val
# because Lightning (correctly) doesn't have a test step during training



hparams['model_hparams'] = {
    'd_model': 128,
    'nhead': 4,
    'num_encoder_layers': 4, 
    'dropout': 0.1,
    'ntoken': 3, # 0,1,pad
    'lr': 3e-5,
    'pad_token_id': hparams['pad_token_id'],
    'weight_decay': 0,
    'backprop_every': hparams['n_automata_steps'],
}


# log all hyperparameters
wandb_logger.log_hyperparams(hparams)


# setup data
ds_train, ds_val, = dataset.make_datasets(hparams['n_train'], hparams['n_val'], n_steps=hparams['n_automata_steps'])
dl_train = data.DataLoader(ds_train, batch_size=hparams['bs'], collate_fn=dataset.collate_fn)
dl_val = data.DataLoader(ds_val, batch_size=hparams['bs'], collate_fn=dataset.collate_fn)

# model
basic_model = model.BPTTTransformer(**hparams['model_hparams'])

# training
trainer = L.Trainer(accelerator='mps', logger=wandb_logger, val_check_interval=hparams['val_check_interval'], 
                    gradient_clip_val=1.0)
trainer.fit(basic_model, dl_train, dl_val)



