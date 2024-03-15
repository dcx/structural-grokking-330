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


        # # models for translating to and from BPTT format

        # encode long input into single token
        self.enc_in = nn.TransformerEncoder(mb.TransformerEncoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        self.dec_in = nn.TransformerDecoder(mb.TransformerDecoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        dec_starter = torch.rand_like(torch.zeros(d_model)) * math.sqrt(d_model)
        self.dec_starter = torch.nn.Parameter(dec_starter) # (d_model,)

        # self.enc_out = nn.TransformerEncoder(mb.TransformerEncoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        #self.linear = nn.Linear(d_model, n_tokens)

        # models for game thinking
        # self.enc_bptt = nn.TransformerEncoder(mb.TransformerEncoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        #self.enc_step1 = nn.TransformerEncoder(mb.TransformerEncoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        #self.dec_bptt = nn.TransformerDecoder(mb.TransformerDecoderLayer(d_model=d_model, nhead=n_enc_heads*2, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)


        # models



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
        x_enc_in = self.enc_in(x_enc_long) # (1, bs*seq_len, d_model)

        x_enc_step1 = self.enc_step1(x_enc_in) # (1, bs*seq_len, d_model)
        y_pred_cur = self.linear(x_enc_step1) # (1, bs*seq_len, n_tokens)
        y_pred_cur = y_pred_cur.reshape(seq_len, bs, -1) # (seq_len, bs, n_tokens)
        y_pred[0] = y_pred_cur

        # do recursive thinking
        x_enc_bptt = x_enc_step1 # (1, bs*seq_len, d_model)
        for i in range(1,self.n_bptt):
            x_enc_bptt = self.dec_bptt(x_enc_bptt, x_enc_in) # (1, bs*seq_len, d_model)
            # x_enc_bptt = self.enc_bptt(x_enc_long) # (1, bs*seq_len, d_model)
            # x_enc_out = self.enc_out(x_enc_bptt) # (1, bs*seq_len, d_model)
            y_pred_cur = self.linear(x_enc_bptt) # (1, bs*seq_len, n_tokens)
            y_pred_cur = y_pred_cur.reshape(seq_len, bs, -1) # (seq_len, bs, n_tokens)
            y_pred[i] = y_pred_cur

        return y_pred # (n_bptt, seq_len, bs, n_tokens)


# Define entire system as a LightningModule
class S2Transformer(L.LightningModule):
    def __init__(self, d_model, n_enc_heads, n_enc_layers, n_tokens, lr, weight_decay, pad_token_id, predictive, dropout, n_bptt, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.d_model = d_model
        self.lr = lr
        self.wd = weight_decay
        self.pad_token_id = pad_token_id
        self.predictive = predictive
        self.metrics = {}
        self.dropout = dropout
        self.n_bptt = n_bptt

        # # freeze s1_model
        # self.s1_model = s1_model
        # for param in self.s1_model.parameters():
        #     param.requires_grad = False

        # encode: long input -> single token
        self.enc_in = nn.TransformerEncoder(mb.TransformerEncoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        self.dec_in = nn.TransformerDecoder(mb.TransformerDecoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        dec_in_starter = torch.rand_like(torch.zeros(d_model)) * math.sqrt(d_model)
        self.dec_in_starter = torch.nn.Parameter(dec_in_starter) # (d_model,)

        # decode: single token -> long output
        self.dec_out = nn.TransformerDecoder(mb.TransformerDecoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        dec_out_starter = torch.rand_like(torch.zeros(d_model)) * math.sqrt(d_model)
        self.dec_out_starter = torch.nn.Parameter(dec_out_starter) # (d_model,)
        self.dec_out_linear = nn.Linear(d_model, n_tokens)

        # metrics
        for mode in ['train', 'val']:
            setattr(self, f'{mode}_x_vae_acc', torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_tokens, ignore_index=pad_token_id))
            self.metrics[f'{mode}_x_vae_acc'] = getattr(self, f'{mode}_x_vae_acc')

        self.init_weights()

        # # s2 model
        # self.s2_model = S2Model(self.n_bptt, d_model, n_enc_heads, n_enc_layers, n_tokens, lr, weight_decay, pad_token_id, dropout)
        # self.init_weights()

        # # metrics
        # for mode in ['train', 'val']: # hack: metrics must be on self or Lightning doesn't handle their devices correctly
        #     for i in range(self.n_bptt):
        #         setattr(self, f'{mode}_pred_acc_b{i:02}', torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_tokens, ignore_index=pad_token_id))
        #         self.metrics[f'{mode}_pred_acc_b{i:02}'] = getattr(self, f'{mode}_pred_acc_b{i:02}')
            


    def init_weights(self) -> None:
        initrange = 0.1
        self.dec_out_linear.bias.data.zero_()
        self.dec_out_linear.weight.data.uniform_(-initrange, initrange)


    def model_step(self, batch, batch_idx, mode='train'):
        x, y = batch # (seq_len, bs)
        padding_mask = (x==self.pad_token_id) # (bs, seq_len)
        x, y = x.to(torch.long), y.to(torch.long) # do conversion on GPU (mem bottleneck)

        # encode long input into single token
        x_enc_seq = self.enc_in(x, padding_mask=padding_mask) # (seq_len, bs, d_model)
        x_enc = self.dec_in(self.dec_in_starter, x_enc_seq, memory_key_padding_mask=padding_mask) # (1, bs, d_model)

        # decode: single token into long output (teacher forcing)
        x_pred_in = torch.cat([
            self.dec_out_starter.unsqueeze(1).repeat(1, x_enc_seq.shape[1], 1), # (1, bs, *)
            x_enc_seq[:-1] # (seq_len-1, bs, d_model),
        ], dim=0) # (seq_len, bs, d_model)
        x_pred_padding_mask = torch.cat([
            torch.zeros(x_enc_seq.shape[1], 1, dtype=torch.bool, device=x_enc_seq.device), # (bs, 1)
            padding_mask[1:] # (bs, seq_len-1)
        ], dim=1) # (bs, seq_len)
        x_dec_seq = self.dec_out(x_pred_in, x_enc, tgt_key_padding_mask=x_pred_padding_mask) # (seq_len, bs, d_model)

        x_dec_seq = x_dec_seq.permute(1,0,2) # (bs, seq_len, d_model)
        x_pred_out = self.dec_out_linear(x_dec_seq) # (bs, seq_len, n_tokens)

        # loss
        x_pred_out_ce = x_pred_out.permute(0,2,1) # (bs, n_tokens, seq_len)
        loss = F.cross_entropy(x_pred_out_ce, x.T, ignore_index=self.pad_token_id)
        self.log(f"{mode}_pred_loss", loss)

        # accuracy
        x_hat = torch.argmax(x_pred_out, dim=2) # (bs, seq_len)
        acc = self.metrics[f'{mode}_x_vae_acc'](x_hat, x.T)
        self.log(f"{mode}_x_vae_acc", acc)

        return loss


        




        # # Predict using S2
        # y_pred = self.s2_model(x_enc) # (n_bptt, seq_len, bs, n_tokens)


        # Encode full input into single token



        # # Calculate loss

        # # loss
        # n_bptt, seq_len, bs, n_tokens = y_pred.shape
        # y_pred = y_pred.permute(0,2,3,1) # (n_bptt, bs, n_tokens, seq_len)

        # ce_y = y.repeat(self.n_bptt, 1) # (n_bptt*bs, seq_len)
        # ce_y_pred = y_pred.reshape(-1, n_tokens, seq_len) # (n_bptt*bs, n_tokens, seq_len)
        # loss = F.cross_entropy(ce_y_pred, ce_y, ignore_index=self.pad_token_id)
        # #loss = F.cross_entropy(y_pred[-1], y, ignore_index=self.pad_token_id)
        # self.log(f"{mode}_pred_loss", loss) 

        # # accuracy: compare first-vs-last BPTT step
        # y_hat_allsteps = torch.argmax(y_pred, dim=2) # (n_bptt,bs,seq_len)
        # for i in range(self.n_bptt):
        #     acc = self.metrics[f'{mode}_pred_acc_b{i:02}'](y_hat_allsteps[i], y)
        #     self.log(f"{mode}_pred_acc_b{i:02}", acc)
        #     if i > 0:
        #         self.log(f"{mode}_pred_acc_b{i:02}_delta", acc - prev_acc)
        #     elif i == 0:
        #         first_acc = acc
        #     prev_acc = acc
        # self.log(f"{mode}_pred_acc_full_delta", acc - first_acc)

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




