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
        # self.enc_out = nn.TransformerEncoder(mb.TransformerEncoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        self.linear = nn.Linear(d_model, n_tokens)

        # models for game thinking
        # self.enc_bptt = nn.TransformerEncoder(mb.TransformerEncoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        self.enc_step1 = nn.TransformerEncoder(mb.TransformerEncoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        self.dec_bptt = nn.TransformerDecoder(mb.TransformerDecoderLayer(d_model=d_model, nhead=n_enc_heads*2, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)


        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        """
        x_enc: (seq_len, bs, d_model)
        returns y_pred: (n_bptt, seq_len, bs, n_tokens)

        TODO: Why is there no pad mask?
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
    def __init__(self, s1_model, d_model, n_enc_heads, n_enc_layers, n_tokens, lr, weight_decay, pad_token_id, predictive, dropout, n_bptt, **kwargs):
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
        self.n_tokens = n_tokens

        # freeze s1_model
        self.s1_model = s1_model
        for param in self.s1_model.parameters():
            param.requires_grad = False

        # s2 model
        self.s2_model = S2Model(self.n_bptt, d_model, n_enc_heads, n_enc_layers, n_tokens, lr, weight_decay, pad_token_id, dropout)
        self.init_weights()

        # metrics
        for mode in ['train', 'val']: # hack: metrics must be on self or Lightning doesn't handle their devices correctly
            for i in range(self.n_bptt):
                setattr(self, f'{mode}_pred_acc_b{i:02}', torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_tokens, ignore_index=pad_token_id))
                self.metrics[f'{mode}_pred_acc_b{i:02}'] = getattr(self, f'{mode}_pred_acc_b{i:02}')


    def init_weights(self) -> None:
        pass

    def model_step(self, batch, batch_idx, mode='train'):
        x, y = batch # (bs, seq_len)
        padding_mask = (x==self.pad_token_id) # (bs, seq_len)
        x, y = x.to(torch.long), y.to(torch.long) # do conversion on GPU (mem bottleneck)

        # Encode using frozen S1
        x_emb, x_enc = self.s1_model.forward_enc(x, padding_mask=padding_mask) # (seq_len, bs, d_model)
        seq_len, bs, d_model = x_enc.shape

        # Predict using frozen S1
        x_enc_long = x_enc.reshape(1, -1, d_model) # (1, bs*seq_len, d_model)
        y_pred_enc_long = self.s1_model.pred_enc(x_enc_long) # (1, bs*seq_len, d_model)
        y_pred = self.s1_model.pred_linear(y_pred_enc_long) # (1, bs*seq_len, n_tokens)
        y_pred = y_pred.reshape(seq_len, bs, -1) # (seq_len, bs, n_tokens)
        y_pred = y_pred.permute(1, 0, 2) # (bs, seq_len, n_tokens)

        # Use calibration to focus on low-confidence predictions
        y_prob = F.softmax(y_pred, dim=2) # (bs, seq_len, n_tokens)
        y_maxprob, y_hat = torch.max(y_prob, dim=2) # (bs, seq_len)

        # grab anything < 0.95
        threshold_conf, threshold_trim = 0.95, 0.125
        focus_mask = torch.logical_and(y_maxprob < threshold_conf, ~padding_mask) # (bs, seq_len)
        n_focus = focus_mask.sum()

        # setup s2 inputs: trim into fixed shape
        bs_s2 = int(bs*seq_len*threshold_trim)
        lc_x_enc = torch.zeros(bs_s2, d_model, device=x_enc.device) + self.pad_token_id # (bs_s2, d_model)
        lc_y_pred_enc = torch.zeros(bs_s2, d_model, device=x_enc.device) + self.pad_token_id # (bs_s2, d_model)
        lc_y_pred = torch.zeros(bs_s2, self.n_tokens, device=x_enc.device) + self.pad_token_id # (bs_s2, n_tokens)
        lc_y = torch.zeros(bs_s2, dtype=torch.long, device=x_enc.device) + self.pad_token_id # (bs_s2,)

        lc_x_enc[:] = x_enc.transpose(0,1)[focus_mask][:bs_s2] # (bs_s2, d_model)
        lc_y_pred_enc[:] = y_pred_enc_long.reshape(bs,seq_len,-1)[focus_mask][:bs_s2] # (bs_s2, d_model)
        lc_y_pred[:] = y_pred[focus_mask][:bs_s2] # (bs_s2, n_tokens)
        lc_y[:] = y[focus_mask][:bs_s2] # (bs_s2,)
        lc_mask = lc_y == self.pad_token_id # (bs_s2,)

        # log stats for low-confidence subset
        lc_y_prob = F.softmax(lc_y_pred[:bs_s2], dim=1)
        lc_y_prob_maxprob, lc_y_pred = lc_y_prob.max(dim=1)
        lc_y_prob_avg_maxprob = lc_y_prob_maxprob.mean()
        subset_acc = (lc_y_pred == lc_y)[:n_focus].float().mean()
        self.log(f"{mode}_s1_lowconf_acc", subset_acc)
        self.log(f"{mode}_s1_lowconf_conf", lc_y_prob_avg_maxprob)
    
        # reshape to fit s2 model
        lc_x_enc = lc_x_enc.unsqueeze(0) # (1, bs_s2, d_model)
        lc_y_pred_enc = lc_y_pred_enc.unsqueeze(0) # (1, bs_s2, d_model)
        lc_y_pred = lc_y_pred.unsqueeze(0) # (1, bs_s2, n_tokens)
        lc_y = lc_y.unsqueeze(1) # (bs_s2,1)
        lc_mask = lc_mask.unsqueeze(1) # (bs_s2,1)

        # Predict using S2
        s2_y_pred = self.s2_model(lc_x_enc) # (n_bptt, 1, bs_s2, n_tokens)
        n_bptt, _, bs_s2, n_tokens = s2_y_pred.shape
        s2_y_pred = s2_y_pred.permute(0,2,3,1) # (n_bptt, bs_s2, n_tokens, 1)

        s2_ce_y = lc_y.repeat(self.n_bptt, 1) # (n_bptt*bs_s2, 1)
        s2_ce_y_pred = s2_y_pred.reshape(-1, n_tokens, 1) # (n_bptt*bs_s2, n_tokens, 1)
        loss = F.cross_entropy(s2_ce_y_pred, s2_ce_y, ignore_index=self.pad_token_id)
        self.log(f"{mode}_s2_pred_loss", loss) 

        # accuracy: compare first-vs-last BPTT step
        s2_y_hat_allsteps = torch.argmax(s2_y_pred, dim=2) # (n_bptt,bs,seq_len)
        for i in range(self.n_bptt):
            acc = self.metrics[f'{mode}_pred_acc_b{i:02}'](s2_y_hat_allsteps[i], lc_y)
            self.log(f"{mode}_pred_acc_b{i:02}", acc)
            if i > 0:
                self.log(f"{mode}_pred_acc_b{i:02}_delta", acc - prev_acc)
            elif i == 0:
                first_acc = acc
            prev_acc = acc
        self.log(f"{mode}_pred_acc_full_delta", acc - first_acc)

        # recombine low-conf with full set and report overall accuracy
        s1_acc = torch.sum(torch.logical_and(y_hat == y, ~padding_mask)) / torch.sum(~padding_mask)
        self.log(f"{mode}_s1_acc", s1_acc)

        n_edits = min(n_focus, bs_s2)
        edit_segment = y_pred[focus_mask] # (n_focus, n_tokens)
        edit_segment[:bs_s2] = s2_y_pred[-1].squeeze()[:n_edits]
        y_pred[focus_mask] = edit_segment

        #y_pred[focus_mask][:bs_s2] = s2_y_pred[-1].squeeze() # (bs, seq_len, n_tokens)
        cmb_y_prob = F.softmax(y_pred, dim=2) # (bs, seq_len, n_tokens)
        cmb_y_prob_maxprob, cmb_y_pred = cmb_y_prob.max(dim=2) # (bs, seq_len)
        combined_acc = torch.sum(torch.logical_and(cmb_y_pred == y, ~padding_mask)) / torch.sum(~padding_mask)
        self.log(f"{mode}_s1s2_acc", combined_acc)
        self.log(f"{mode}_s1s2_acc_gain", combined_acc - s1_acc)
        
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




