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
    'bs': 16,
    'num_workers': 8,
    'pad_token_id': 30,

    # dataset
    'csv_file': 'prototyping/system2-data/test-5k-6-8.csv',
    'use_cur_action': True,
    'use_cur_action_result': True,
    'use_next_action': True,
    'val_check_interval': 100, # in steps
    'holdout_trees_frac': 0.15,
    'train_frac': 0.99,
    'val_frac': 0.01, # currently ignored
    'test_max_items': 2000,     
}
# reminder: unlike main framework, here we are plugging test into val
# because Lightning (correctly) doesn't have a test step during training



hparams['model_hparams'] = {
    'd_model': 512,
    'nhead': 8,
    'num_encoder_layers': 6, 
    'dropout': 0.1,
    'ntoken': 32,
    'lr': 3e-4,
    'pad_token_id': hparams['pad_token_id'],
    'weight_decay': 0,
}


# log all hyperparameters
wandb_logger.log_hyperparams(hparams)




# setup data
ds_train, ds_val, ds_test = dataset.make_datasets(
    hparams['csv_file'],
    holdout_trees_frac=hparams['holdout_trees_frac'],
    train_frac=hparams['train_frac'], val_frac=hparams['val_frac'],
    test_max_items=hparams['test_max_items'],
    use_cur_action=hparams['use_cur_action'], 
    use_cur_action_result=hparams['use_cur_action_result'], 
    use_next_action=hparams['use_next_action'])
dl_train = data.DataLoader(ds_train, batch_size=hparams['bs'], num_workers=hparams['num_workers'], collate_fn=dataset.collate_fn)
dl_val = data.DataLoader(ds_val, batch_size=hparams['bs'], num_workers=hparams['num_workers'], collate_fn=dataset.collate_fn)
dl_test = data.DataLoader(ds_test, batch_size=hparams['bs'], num_workers=hparams['num_workers'], collate_fn=dataset.collate_fn)

# model
basic_model = model.PlanTransformer(**hparams['model_hparams'])

# training
trainer = L.Trainer(logger=wandb_logger, val_check_interval=hparams['val_check_interval'])
trainer.fit(basic_model, dl_train, dl_test)





