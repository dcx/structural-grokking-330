import torch, torch.nn as nn, torch.utils.data as data, torch.nn.functional as F
import lightning as L
import os
import pandas as pd
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

import model, dataset

# wandb setup
wandb_logger = WandbLogger(dir='/nlp/scr/ananjan/wandb', log_model="all")

# GPU setup
torch.set_float32_matmul_precision('highest')


# hyperparameters
hparams = {
    'bs': 16,
    'pad_token_id': 30, # TODO: Attach to dataset.py
    'sep_token_id': 31, # TODO: Attach to dataset.py

    # dataset
    'csv_file': '/nlp/scr/ananjan/amlif-data/amlif-10k.csv', # '../data/amlif-50k.csv',
    'use_cur_action': False,
    'use_cur_action_result': False,
    'use_next_action': False,
    'val_check_interval': 2500, # in steps
    'holdout_trees_frac': 0.15,
    'train_frac': 0.99,
    'val_frac': 0.01, # currently ignored
    'train_max_items': None, # debug: tiny subset for faster load. leave as None when not used
    'val_max_items': 2500,
    'test_max_items': 2500, 
}
# reminder: unlike main framework, here we are plugging test into val
# because Lightning (correctly) doesn't have a test step during training



hparams['model_hparams'] = {
    'd_model': 384,
    'n_enc_heads': 8,
    'n_enc_layers': 6, 
    'dropout': 0.1,
    'n_unique_tokens': 32,
    'n_output_tokens': 32,
    'lr': 1e-4,
    'pad_token_id': hparams['pad_token_id'],
    'weight_decay': 0.1, # https://arxiv.org/pdf/2307.06435.pdf table 6
    'max_steps': 10, # upper bound on BPTT steps when randomly sampling
}


# log all hyperparameters
wandb_logger.log_hyperparams(hparams)


# setup data
ds_train, ds_val, ds_test = dataset.make_datasets(
    hparams['csv_file'],
    holdout_trees_frac=hparams['holdout_trees_frac'],
    train_frac=hparams['train_frac'], val_frac=hparams['val_frac'],
    train_max_items=hparams['train_max_items'],
    test_max_items=hparams['test_max_items'],
    val_max_items=hparams['val_max_items'],
    use_cur_action=hparams['use_cur_action'], 
    use_cur_action_result=hparams['use_cur_action_result'], 
    use_next_action=hparams['use_next_action'])

dl_train = data.DataLoader(ds_train, batch_size=hparams['bs'], collate_fn=dataset.collate_fn, shuffle=True, pin_memory=True)
dl_val = data.DataLoader(ds_val, batch_size=hparams['bs'], collate_fn=dataset.collate_fn, shuffle=True, pin_memory=True)

# model
basic_model = model.BPTTTransformer(**hparams['model_hparams'])

# callbacks
checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
    monitor='train_loss',
    every_n_train_steps=2500,
    dirpath='/nlp/scr/ananjan/checkpoints',
    filename='model-10k-{epoch:02d}-{step:08d}',
    save_top_k=1,
    mode='min',
)
lr_monitor_callback = L.pytorch.callbacks.LearningRateMonitor(logging_interval='step')

# training
trainer = L.Trainer(accelerator='gpu', logger=wandb_logger, val_check_interval=hparams['val_check_interval'], 
                    gradient_clip_val=1.0, callbacks=[checkpoint_callback, lr_monitor_callback])
trainer.fit(basic_model, dl_train, dl_val)



