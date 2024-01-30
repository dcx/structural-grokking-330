# 41-vae-shared-decoder
# This is just like 40-vae, except we prep S1 for next-word prediction:
# we can now parallelize S1 using an encoder. We set up chess.

import torch, torch.nn as nn, torch.utils.data as data, torch.nn.functional as F
import lightning as L
import os
import pandas as pd
from pytorch_lightning.loggers import WandbLogger

import model, dataset

# wandb setup
wandb_logger = WandbLogger(log_model=False)

# GPU setup
torch.set_float32_matmul_precision('medium')


# hyperparameters
hparams = {
    'bs': 16,
    'pad_token_id': dataset.pad_token_id,

    # dataset
    'val_check_interval': 1500, # in steps
    'n_train': 5000000,     
    'n_val': 256,     
}
# reminder: unlike main framework, here we are plugging test into val
# because Lightning (correctly) doesn't have a test step during training


hparams['model_hparams'] = {
    'd_model': 128,
    'n_enc_heads': 8,
    'n_enc_layers': 6, 
    'n_tokens': len(dataset.stoi), # 32 for chess
    'lr': 5e-4,
    'weight_decay': 0.1,
    'pad_token_id': hparams['pad_token_id'],
    'predictive': True,
    'dropout': 0.0,
}


# log all hyperparameters
wandb_logger.log_hyperparams(hparams)


# setup data
ds_train, ds_val = dataset.make_datasets(hparams['n_train'], hparams['n_val'])
collate_fn = dataset.collate_fn
sampler_train = dataset.SeqLenAwareBatchSampler(ds_train, hparams['bs'], shuffle=True)
sampler_val = dataset.SeqLenAwareBatchSampler(ds_val, hparams['bs'], shuffle=True)
dl_train = data.DataLoader(ds_train, collate_fn=collate_fn, pin_memory=True, batch_sampler=sampler_train, num_workers=4)
dl_val = data.DataLoader(ds_val, collate_fn=collate_fn, pin_memory=True, batch_sampler=sampler_val, num_workers=4)

# model
# basic_model = model.S1Transformer(**hparams['model_hparams'])
basic_model = model.S1Transformer.load_from_checkpoint("../checkpoints/epoch=1-step=17625.ckpt")

# checkpointing
checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
    monitor='train_vae_loss',
    every_n_train_steps=2500,
    dirpath='../checkpoints',
    filename='model-{epoch:02d}-{step:08d}',
    save_top_k=3,
    mode='min',
)

# training
trainer = L.Trainer(accelerator='gpu', logger=wandb_logger, val_check_interval=hparams['val_check_interval'], 
                    gradient_clip_val=1.0, callbacks=[checkpoint_callback])
trainer.fit(basic_model, dl_train, dl_val)



