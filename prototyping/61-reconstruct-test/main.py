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
    'bs': 128,
    'pad_token_id': dataset.pad_token_id,
    'cpu_procs': 8,

    # dataset
    'val_check_interval': 150, # in steps
    'n_train': None, # 12800,     
    'n_val': 512,     

    # model
    'pt1_checkpoint': None, # "../checkpoints/model-1711078733-epoch=00-step=00012500.ckpt", # "../checkpoints/model-epoch=01-step=00567500.ckpt", # None for fresh train
    'pt2_mode': False, #True, # True,
}
# reminder: unlike main framework, here we are plugging test into val
# because Lightning (correctly) doesn't have a test step during training


hparams['model_hparams'] = {
    'd_model': 768,
    'n_enc_heads': 12,
    'n_enc_layers': 8, 
    'n_tokens': len(dataset.stoi), # 32 for chess
    'lr': 1e-4,
    'weight_decay': 0.1,
    'pad_token_id': hparams['pad_token_id'],
    'predictive': True,
    'dropout': 0.0,
    'n_bptt': 4,
}


# log all hyperparameters
wandb_logger.log_hyperparams(hparams)


# setup data
ds_train, ds_val = dataset.make_datasets(hparams['n_train'], hparams['n_val'], num_proc=hparams['cpu_procs'])
collate_fn = dataset.collate_fn
#sampler_train = dataset.SeqLenAwareBatchSampler(ds_train, hparams['bs'], shuffle=True)
#sampler_val = dataset.SeqLenAwareBatchSampler(ds_val, hparams['bs'], shuffle=True)
dl_train = data.DataLoader(ds_train, collate_fn=collate_fn, pin_memory=True, num_workers=hparams['cpu_procs'], batch_size=hparams['bs'], shuffle=True)
dl_val = data.DataLoader(ds_val, collate_fn=collate_fn, pin_memory=True, num_workers=hparams['cpu_procs'], batch_size=hparams['bs'], shuffle=False)

# model

if hparams['pt1_checkpoint'] is None:
    pt1_model = model_s2.S2Transformer(**hparams['model_hparams']) # train on depths 2-6 so s1 can learn depths 2-3 (but not 4-6)
else:
    pt1_model = model_s2.S2Transformer.load_from_checkpoint(hparams['pt1_checkpoint'])

if hparams['pt2_mode']:
    main_model = model_s2.S2TransformerP2(pt1_model, **hparams['model_hparams']) # train on depths 4-6 so every item in every batch is hard
else:
    main_model = pt1_model

# checkpointing
time_secs = int(time.time())
fname_prefix = f"model-{time_secs}"
checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
    monitor='step',
    every_n_train_steps=1250,
    dirpath='../checkpoints',
    filename=fname_prefix+'-{epoch:02d}-{step:08d}',
    save_top_k=3,
    mode="max",
)

# training
trainer = L.Trainer(accelerator='gpu', logger=wandb_logger, val_check_interval=hparams['val_check_interval'], 
                    gradient_clip_val=1.0, callbacks=[checkpoint_callback], devices=[0])




# LR finder
# from lightning.pytorch.tuner import Tuner
# tuner = Tuner(trainer)
# lr_finder = tuner.lr_find(model)
# print(lr_finder.results)
# # Plot with
# fig = lr_finder.plot(suggest=True)
# fig.show()
# new_lr = lr_finder.suggestion()
# print(new_lr)


trainer.fit(main_model, dl_train, dl_val)



