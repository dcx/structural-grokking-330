import torch, torch.nn as nn, torch.utils.data as data, torch.nn.functional as F
import lightning as L
import os
import pandas as pd
from pytorch_lightning.loggers import WandbLogger

import model, dataset

# hyperparameters
bs = 64
num_workers = 8
pad_token_id = 30

model_hparams = {
    'd_model': 512,
    'nhead': 8,
    'num_encoder_layers': 6, 
    'dropout': 0.1,
    'ntoken': 32,
    'lr': 1e-4,
    'pad_token_id': pad_token_id,
    'weight_decay': 0.01,
}

val_check_interval = 2500 # in steps


# setup data
ds = dataset.PlanDataset('prototyping/system2-data/test-100k.csv')
train, val = data.random_split(ds, [0.9, 0.1])
dl_train = data.DataLoader(train, batch_size=bs, num_workers=num_workers, collate_fn=dataset.collate_fn)
dl_val = data.DataLoader(val, batch_size=bs, num_workers=num_workers, collate_fn=dataset.collate_fn)

# model
basic_model = model.PlanTransformer(**model_hparams)

# training
wandb_logger = WandbLogger(log_model="all")
trainer = L.Trainer(logger=wandb_logger, val_check_interval=val_check_interval)
trainer.fit(basic_model, dl_train, dl_val)





