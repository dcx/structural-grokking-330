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

# required to run bptt over in->loop->out transfomers - not clear why
# https://stackoverflow.com/questions/77343471/pytorch-cuda-error-invalid-configuration-argument
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

parser = argparse.ArgumentParser()
parser.add_argument("--bs", type=int, default=4, help="Batch size")
parser.add_argument("--lr_s1", type=float, default=1e-5, help="Learning rate for S1")
parser.add_argument("--lr_s2", type=float, default=1e-5, help="Learning rate for S2")
parser.add_argument("--n_train", type=int, default=1000000, help="Number of training samples")
parser.add_argument("--n_val", type=int, default=256, help="Number of validation samples")
parser.add_argument("--val_check_interval", type=int, default=1500, help="Validation check interval")
parser.add_argument("--vae_checkpoint", type=str, default='../checkpoints/model-epoch=01-step=00567500.ckpt', help="Path to VAE checkpoint")
parser.add_argument("--s2_checkpoint", type=str, default=None, help="Path to checkpoint for resuming full train")
parser.add_argument("--cpu_procs", type=int, default=6, help="Number of CPU processes")
# parser.add_argument("--cuda_id", type=int, default=0, help="GPU to use")
parser.add_argument("--n_bptt", type=int, default=16, help="Number of S2 inner loops")
parser.add_argument("--bptt_every_loop", action="store_true", help="Backprop into each S2 loop, instead of just the last one?")
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
    'd_model': 512,
    'n_enc_heads': 8,
    'n_enc_layers': 8,
    'n_tokens': len(dataset.stoi), # 32 for chess
    'lr_s1': hparams['lr_s1'],
    'lr_s2': hparams['lr_s2'],
    'weight_decay': 0.1,
    'pad_token_id': dataset.pad_token_id,
    'predictive': True,
    'dropout': 0.0,
    'n_bptt': hparams['n_bptt'],
    'max_bs': hparams['bs'],
    'max_seq_len': 512,
    'bptt_every_loop': hparams['bptt_every_loop'],
}

# setup data
ds_train, ds_val = dataset.make_datasets(hparams['n_train'], hparams['n_val'], num_proc=hparams['cpu_procs'])
collate_fn = dataset.collate_fn
#sampler_train = dataset.SeqLenAwareBatchSampler(ds_train, hparams['bs'], shuffle=True)
#sampler_val = dataset.SeqLenAwareBatchSampler(ds_val, hparams['bs'], shuffle=True)
dl_train = data.DataLoader(ds_train, collate_fn=collate_fn, pin_memory=True, num_workers=hparams['cpu_procs'], batch_size=hparams['bs'], shuffle=True) # batch_sampler=sampler_train, 
dl_val = data.DataLoader(ds_val, collate_fn=collate_fn, pin_memory=True, num_workers=hparams['cpu_procs'], batch_size=hparams['bs'], shuffle=False) # batch_sampler=sampler_val, 

# model

if hparams['vae_checkpoint'] is None:
    vae_model = model.S1Transformer(**hparams['model_hparams'])
else:
    vae_model = model.S1Transformer.load_from_checkpoint(hparams['vae_checkpoint']) # .to(f"cuda:{hparams['cuda_id']}")

main_model = model_s2.S2Transformer(vae_model, **hparams['model_hparams'])
if hparams['s2_checkpoint']:
    ckpt_path = hparams['s2_checkpoint']
else:
    ckpt_path = None


# log all hyperparameters, gradients and model topology
wandb_logger.log_hyperparams(hparams)
wandb_logger.watch(main_model, log_freq=200)


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
                    callbacks=[checkpoint_callback], devices=1) # devices=[hparams['cuda_id']])
trainer.fit(main_model, dl_train, dl_val, ckpt_path=ckpt_path)



