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
    'bs': 64,
    'pad_token_id': 4,

    # dataset
    'val_check_interval': 150, # in steps
    'n_train': 500000,     
    'n_val': 500,     

    'n_automata_steps': 10,
}
# reminder: unlike main framework, here we are plugging test into val
# because Lightning (correctly) doesn't have a test step during training



hparams['model_hparams'] = {
    'd_model': 512,
    'n_enc_heads': 8,
    'n_enc_layers': 6, 
    'n_unique_tokens': 5, # 0,1,ca_rule_0,ca_rule_1,pad
    'n_output_tokens': 2, # 0,1
    'lr': 1e-4,
    'pad_token_id': hparams['pad_token_id'],
    'weight_decay': 0.1,
    'max_steps': hparams['n_automata_steps'],
    'pad_token_id': hparams['pad_token_id'],
    'predictive': True,
    'dropout': 0.0,
}


# log all hyperparameters
wandb_logger.log_hyperparams(hparams)


# setup data
ds_train = dataset.ToyDataset(n_examples=hparams['n_train'], len_from=8, len_to=16, n_steps=hparams['n_automata_steps'], ca_rule_no=30)
ds_val  =  dataset.ToyDataset(n_examples=hparams['n_val'], len_from=8, len_to=16, n_steps=hparams['n_automata_steps'], ca_rule_no=30)
collate_fn = dataset.make_collate_fn(hparams['pad_token_id'])
dl_train = data.DataLoader(ds_train, batch_size=hparams['bs'], collate_fn=collate_fn)
dl_val = data.DataLoader(ds_val, batch_size=hparams['bs'], collate_fn=collate_fn)

# model
basic_model = model.S1Transformer(**hparams['model_hparams'])
# basic_model = model.BPTTTransformer.load_from_checkpoint("lightning_logs/15zxa4h3/checkpoints/epoch=18-step=57850.ckpt")

# checkpointing
checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
    monitor='train_loss',
    every_n_train_steps=2500,
    dirpath='../checkpoints',
    filename='model-{epoch:02d}-{step:08d}',
    save_top_k=3,
    mode='min',
)

# training
trainer = L.Trainer(accelerator='gpu', logger=wandb_logger, val_check_interval=hparams['val_check_interval'], 
                    gradient_clip_val=1.0)
trainer.fit(basic_model, dl_train, dl_val)



