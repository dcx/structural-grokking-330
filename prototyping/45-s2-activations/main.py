# 41-vae-shared-decoder
# This is just like 40-vae, except we prep S1 for next-word prediction:
# we can now parallelize S1 using an encoder. We set up chess.

import torch, torch.nn as nn, torch.utils.data as data, torch.nn.functional as F
import lightning as L
import os, time
import pandas as pd
from pytorch_lightning.loggers import WandbLogger
import argparse

import model, model_s2, dataset

# wandb setup
wandb_logger = WandbLogger(log_model=False)

# GPU setup
torch.set_float32_matmul_precision('medium')


parser = argparse.ArgumentParser()
parser.add_argument("--bs", type=int, default=32, help="Batch size")
parser.add_argument("--n_train", type=int, default=1000000, help="Number of training samples")
parser.add_argument("--n_val", type=int, default=256, help="Number of validation samples")
parser.add_argument("--val_check_interval", type=int, default=1500, help="Validation check interval")
parser.add_argument("--vae_checkpoint", type=str, default=None, help="Path to VAE checkpoint")
parser.add_argument("--s2_checkpoint", type=str, default=None, help="Path to checkpoint for resuming full train")
parser.add_argument("--cpu_procs", type=int, default=6, help="Number of CPU processes")
parser.add_argument("--cuda_id", type=int, default=0, help="GPU to use")
parser.add_argument("--n_s2_loops", type=int, default=2, help="Number of S2 inner loops")
hparams = vars(parser.parse_args())

# # hyperparameters
# hparams = {
#     'bs': 32,
#     'pad_token_id': dataset.pad_token_id,
#     'cpu_procs': 6,

#     # dataset
#     'val_check_interval': 1500, # in steps
#     'n_train': 1000000,     
#     'n_val': 256,     

#     # model (strictly speaking, this is the old S1 model, but we're now only using it for the pretrained VAE)
#     'vae_checkpoint': "../checkpoints/model-epoch=01-step=00567500.ckpt", # None for fresh train
#     's2_checkpoint': None # "../checkpoints/model-s2-1707058090-epoch=00-step=00007500.ckpt",
# }
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
    'pad_token_id': dataset.pad_token_id,
    'predictive': True,
    'dropout': 0.0,
    'n_bptt': 1,
    'max_bs': hparams['bs'],
    'max_seq_len': 512,
    'n_s2_loops': hparams['n_s2_loops'],
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
    vae_model = model.S1Transformer.load_from_checkpoint(hparams['vae_checkpoint']).to(f"cuda:{hparams['cuda_id']}")

main_model = model_s2.S2Transformer(vae_model, **hparams['model_hparams'])
if hparams['s2_checkpoint']:
    ckpt_path = hparams['s2_checkpoint']
else:
    ckpt_path = None


# checkpointing
time_secs = int(time.time())
fname_prefix = f"model-s2-{time_secs}"
checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
    monitor='step',
    mode='max',
    every_n_train_steps=2500,
    dirpath='../checkpoints',
    filename=fname_prefix+'-{epoch:02d}-{step:08d}',
    save_top_k=3,
)
print(f"Checkpoint identifier: {fname_prefix}")

# training
trainer = L.Trainer(accelerator='gpu', logger=wandb_logger, val_check_interval=hparams['val_check_interval'], 
                    callbacks=[checkpoint_callback], devices=[hparams['cuda_id']])
trainer.fit(main_model, dl_train, dl_val, ckpt_path=ckpt_path)



