import torch, torch.nn as nn, torch.utils.data as data, torch.nn.functional as F
import lightning as L
import math
import torchmetrics
import random
import modelbasics as mb



class S2Model(nn.Module):
    """
    This module is for System 2

    Simple start: Accepts encoded game states from VAE+S1, and predicts the next action.
    """

    def __init__(self, n_bptt, d_model, n_enc_heads, n_enc_layers, n_tokens, lr, weight_decay, pad_token_id, dropout):
        super().__init__()

        self.d_model = d_model
        self.lr = lr
        self.wd = weight_decay
        self.pad_token_id = pad_token_id
        self.dropout = dropout
        self.n_tokens = n_tokens

        self.n_bptt = n_bptt


        # models for translating to and from BPTT format
        self.enc_in = nn.TransformerEncoder(mb.TransformerEncoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        #self.dec_out = nn.TransformerDecoder(mb.TransformerDecoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        self.linear = nn.Linear(d_model, n_tokens)

        # encoder for recursive thinking (BPTT)
        self.enc_bptt = nn.TransformerEncoder(mb.TransformerEncoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        """
        x_enc: (seq_len, bs, d_model)
        returns y_pred: (n_bptt, seq_len, bs, n_tokens)
        """
        seq_len, bs, d_model = x_enc.shape

        # init results tensor
        y_pred = torch.zeros(self.n_bptt, seq_len, bs, self.n_tokens, device=x_enc.device) # (n_bptt, seq_len, bs, n_tokens)

        # each x_enc represents an entire game sequence, so predict with a single token per batch: (1, bs*seq_len, d_model)
        x_enc_long = x_enc.reshape(1, -1, d_model) # (1, bs*seq_len, d_model)

        # encode for BPTT
        x_enc_long = self.enc_bptt(x_enc_long) # (1, bs*seq_len, d_model)
        # do recursive thinking
        for i in range(self.n_bptt):
            x_enc_long = self.enc(x_enc_long) # (1, bs*seq_len, d_model)
            y_pred_cur = self.linear(x_enc_long) # (1, bs*seq_len, n_tokens)
            y_pred_cur = y_pred_cur.reshape(seq_len, bs, -1) # (seq_len, bs, n_tokens)
            y_pred[i] = y_pred_cur

        return y_pred # (n_bptt, seq_len, bs, n_tokens)


# Define entire system as a LightningModule
class S2Transformer(L.LightningModule):
    def __init__(self, s1_model, d_model, n_enc_heads, n_enc_layers, n_tokens, lr, weight_decay, pad_token_id, predictive, dropout):
        super().__init__()
        self.save_hyperparameters()

        self.d_model = d_model
        self.lr = lr
        self.wd = weight_decay
        self.pad_token_id = pad_token_id
        self.predictive = predictive
        self.metrics = {}
        self.dropout = dropout
        self.n_bptt = 1

        # freeze s1_model
        self.s1_model = s1_model
        for param in self.s1_model.parameters():
            param.requires_grad = False

        # s2 model
        self.s2_model = S2Model(self.n_bptt, d_model, n_enc_heads, n_enc_layers, n_tokens, lr, weight_decay, pad_token_id, dropout)
        self.init_weights()

        # metrics
        for mode in ['train', 'val']: # hack: metrics must be on self or Lightning doesn't handle their devices correctly
            setattr(self, f'{mode}_pred_acc', torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_tokens, ignore_index=pad_token_id))
            self.metrics[f'{mode}_pred_acc'] = getattr(self, f'{mode}_pred_acc')
            setattr(self, f'{mode}_pred_acc_bptt0', torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_tokens, ignore_index=pad_token_id))
            self.metrics[f'{mode}_pred_acc_bptt0'] = getattr(self, f'{mode}_pred_acc_bptt0')



    def init_weights(self) -> None:
        pass

    def model_step(self, batch, batch_idx, mode='train'):
        x, y = batch # (bs, seq_len)
        padding_mask = (x==self.pad_token_id) # (bs, seq_len)
        x, y = x.to(torch.long), y.to(torch.long) # do conversion on GPU (mem bottleneck)

        # Encode using S1
        x_emb, x_enc = self.s1_model.forward_enc(x, padding_mask=padding_mask) # (seq_len, bs, d_model)

        # Predict using S2
        y_pred = self.s2_model(x_enc) # (n_bptt, seq_len, bs, n_tokens)


        # Calculate loss

        # loss
        n_bptt, seq_len, bs, n_tokens = y_pred.shape
        y_pred = y_pred.permute(0,2,3,1) # (n_bptt, bs, n_tokens, seq_len)

        #ce_y = y.repeat(self.n_bptt, 1) # (n_bptt*bs, seq_len)
        #ce_y_pred = y_pred.reshape(-1, n_tokens, seq_len) # (n_bptt*bs, n_tokens, seq_len)
        loss = F.cross_entropy(y_pred[-1], y, ignore_index=self.pad_token_id)
        self.log(f"{mode}_pred_loss", loss) 

        # accuracy: measure only on first, last prediction
        y_pred_range = y_pred[-1] # (bs, n_tokens, seq_len)
        y_hat = torch.argmax(y_pred_range, dim=1) # (bs, seq_len)
        acc_p_last = self.metrics[f'{mode}_pred_acc'](y_hat, y)
        self.log(f"{mode}_pred_acc", self.metrics[f'{mode}_pred_acc']) # a = (y[0] != y_hat[0])*~padding_mask[0]

        y_pred_range_p0 = y_pred[0] # (bs, n_tokens, seq_len)
        y_hat_p0 = torch.argmax(y_pred_range_p0, dim=1) # (bs, seq_len)
        acc_p0 = self.metrics[f'{mode}_pred_acc_bptt0'](y_hat_p0, y)
        self.log(f"{mode}_pred_acc_bptt0", self.metrics[f'{mode}_pred_acc_bptt0']) # a = (y[0] != y_hat[0])*~padding_mask[0]

        self.log(f"{mode}_pred_s2_delta", acc_p_last - acc_p0)

        return loss

    def training_step(self, batch, batch_idx):
        return self.model_step(batch, batch_idx, mode='train')
    
    def validation_step(self, batch, batch_idx):
        return self.model_step(batch, batch_idx, mode='val')

    def log_and_reset_metrics(self, mode):
        for metric_name, metric in self.metrics.items():
            if metric_name.startswith(mode):
                self.log(f"{metric_name}_epoch", metric.compute())
                metric.reset()

    def on_train_epoch_end(self):
        self.log_and_reset_metrics('train')

    def on_val_epoch_end(self):
        self.log_and_reset_metrics('val')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer




