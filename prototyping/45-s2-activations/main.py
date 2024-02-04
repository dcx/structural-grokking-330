# 41-vae-shared-decoder
# This is just like 40-vae, except we prep S1 for next-word prediction:
# we can now parallelize S1 using an encoder. We set up chess.

import torch, torch.nn as nn, torch.utils.data as data, torch.nn.functional as F
import lightning as L
import os, time
import pandas as pd
from pytorch_lightning.loggers import WandbLogger

import model, model_s2, dataset

# wandb setup
wandb_logger = WandbLogger(log_model=False)

# GPU setup
torch.set_float32_matmul_precision('medium')


# hyperparameters
hparams = {
    'bs': 64,
    'pad_token_id': dataset.pad_token_id,
    'cpu_procs': 6,

    # dataset
    'val_check_interval': 1500, # in steps
    'n_train': 1000000,     
    'n_val': 256,     

    # model (strictly speaking, this is the old S1 model, but we're now only using it for the pretrained VAE)
    'vae_checkpoint': "../checkpoints/model-epoch=01-step=00567500.ckpt", # None for fresh train
    's2_mode': True,
}
# reminder: unlike main framework, here we are plugging test into val
# because Lightning (correctly) doesn't have a test step during training


hparams['model_hparams'] = {
    'd_model': 128,
    'n_enc_heads': 8,
    'n_enc_layers': 6, 
    'n_tokens': len(dataset.stoi), # 32 for chess
    'lr_s1': 1e-4,
    'lr_s2': 1e-4,
    'weight_decay': 0.1,
    'pad_token_id': hparams['pad_token_id'],
    'predictive': True,
    'dropout': 0.0,
    'n_bptt': 1,
    'max_bs': hparams['bs'],
    'max_seq_len': 512,
}


# log all hyperparameters
wandb_logger.log_hyperparams(hparams)


# setup data
ds_train, ds_val = dataset.make_datasets(hparams['n_train'], hparams['n_val'], num_proc=hparams['cpu_procs'])
collate_fn = dataset.collate_fn
sampler_train = dataset.SeqLenAwareBatchSampler(ds_train, hparams['bs'], shuffle=True)
sampler_val = dataset.SeqLenAwareBatchSampler(ds_val, hparams['bs'], shuffle=True)
dl_train = data.DataLoader(ds_train, collate_fn=collate_fn, pin_memory=True, batch_sampler=sampler_train, num_workers=hparams['cpu_procs'])
dl_val = data.DataLoader(ds_val, collate_fn=collate_fn, pin_memory=True, batch_sampler=sampler_val, num_workers=hparams['cpu_procs'])

# model

if hparams['vae_checkpoint'] is None:
    vae_model = model.S1Transformer(**hparams['model_hparams'])
else:
    vae_model = model.S1Transformer.load_from_checkpoint(hparams['vae_checkpoint']).to('cuda:1')

if hparams['s2_mode']:
    main_model = model_s2.S2Transformer(vae_model, **hparams['model_hparams'])
else:
    main_model = vae_model 


# checkpointing
time_secs = int(time.time())
fname_prefix = f"model-s2-{time_secs}"
checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
    monitor='step',
    every_n_train_steps=2500,
    dirpath='../checkpoints',
    filename=fname_prefix+'-{epoch:02d}-{step:08d}',
    save_top_k=3,
    mode='min',
)

# training
trainer = L.Trainer(accelerator='gpu', logger=wandb_logger, val_check_interval=hparams['val_check_interval'], 
                    callbacks=[checkpoint_callback], devices=[1])
trainer.fit(main_model, dl_train, dl_val)



