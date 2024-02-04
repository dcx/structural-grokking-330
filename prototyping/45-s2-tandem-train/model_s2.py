import torch, torch.nn as nn, torch.utils.data as data, torch.nn.functional as F
import lightning as L
import math
import torchmetrics
import random
import modelbasics as mb
import pl_bolts



class S2Model(nn.Module):
    """
    This module is for System 2

    Simple start: Accepts encoded game states from VAE+S1, and predicts the next action.
    """

    def __init__(self, s1_model, n_bptt, d_model, n_enc_heads, n_enc_layers, n_tokens, lr, weight_decay, pad_token_id, dropout):
        super().__init__()

        self.s1_model = s1_model
        self.n_bptt = n_bptt

        self.d_model = d_model
        self.lr = lr
        self.wd = weight_decay
        self.pad_token_id = pad_token_id
        self.dropout = dropout
        self.n_tokens = n_tokens



        dummy_s2_starter = torch.rand_like(torch.zeros(d_model)) * math.sqrt(d_model)
        self.dummy_s2_starter = torch.nn.Parameter(dummy_s2_starter) # (d_model,)

        # # models for translating to and from BPTT format
        self.enc_x = nn.TransformerEncoder(mb.TransformerEncoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        self.dec_s2_1 = nn.TransformerDecoder(mb.TransformerDecoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        self.dec_s2_2 = nn.TransformerDecoder(mb.TransformerDecoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        self.linear_proj = nn.Linear(d_model, d_model)

        self.enc_out = nn.TransformerEncoder(mb.TransformerEncoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        self.linear_out = nn.Linear(d_model, n_tokens)

        # # models for game thinking
        # # self.enc_bptt = nn.TransformerEncoder(mb.TransformerEncoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        # self.enc_step1 = nn.TransformerEncoder(mb.TransformerEncoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        # self.dec_bptt = nn.TransformerDecoder(mb.TransformerDecoderLayer(d_model=d_model, nhead=n_enc_heads*2, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)

        # setup hooks to get activations from each layer of s1_model

        self.s1_model_hooks = []
        self.s1_model_activations = [None] * len(s1_model.pred_enc.layers)
        def get_hook(i):
            def hook(module, input, output):
                self.s1_model_activations[i] = output
            return hook
        for i in range(len(s1_model.pred_enc.layers)):
            self.s1_model_hooks.append(s1_model.pred_enc.layers[i].register_forward_hook(get_hook(i)))

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear_proj.bias.data.zero_()
        self.linear_proj.weight.data.uniform_(-initrange, initrange)

        self.linear_out.bias.data.zero_()
        self.linear_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, x_enc: torch.Tensor, s1_enc: torch.Tensor) -> torch.Tensor:
        """
        x_enc: (seq_len, bs, d_model)
        s1_enc: (seq_len, bs, d_model)
        returns y_pred: (n_bptt, seq_len, bs, n_tokens)

        No padding_mask required because we're not doing any attention over the sequence.
        So cross-enteropy loss will handle ignoring the padding tokens.
        """
        seq_len, bs, d_model = x_enc.shape

        # prep s2 starter input
        s2_state = torch.zeros(2, bs*seq_len, d_model, device=x_enc.device) # (2, bs*seq_len, d_model)
        s2_state[0, :] = self.dummy_s2_starter
        s2_state[1, :] = x_enc.reshape(-1, d_model)

        # prep output tensor
        # y_pred = torch.zeros(self.n_bptt, seq_len, bs, self.n_tokens, device=x_enc.device) # (n_bptt, seq_len, bs, n_tokens)

        for i in range(8):
            # PHASE 1/2: update state
            # make s1 query (frozen model)
            s1_query = s2_state[-1:] # (1, bs*seq_len, d_model)
            s1_resp = self.s1_model.pred_enc(s1_query) # (1, bs*seq_len, d_model)
            s1_acts = torch.cat(self.s1_model_activations, dim=0) # (n_layers, bs*seq_len, d_model)
            # make x decoder input (encode)
            s2_state_enc = torch.zeros(2, bs*seq_len, d_model, device=x_enc.device) # (2, bs*seq_len, d_model)
            s2_state_enc[0, :] = s2_state[0, :]
            s2_state_enc[1, :] = self.enc_x(s2_state[:1]) # (bs*seq_len, d_model)
            # push through decoder: update state
            s2_phase1 = self.dec_s2_1(s2_state_enc, s1_acts) # (2, bs*seq_len, d_model)
            # PHASE 2/2: glance at original x_enc and update state for next cycle
            s2_state = self.dec_s2_2(s2_phase1, x_enc) # (2, bs*seq_len, d_model)

        # make output pred
        y_pred = self.linear_out(s2_phase1[1:]) # (bs*seq_len, n_tokens)

        # shape for output
        y_pred = y_pred.reshape(seq_len, bs, -1).unsqueeze(0) # (1, seq_len, bs, n_tokens)
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
        self.s2_model = S2Model(s1_model, self.n_bptt, d_model, n_enc_heads, n_enc_layers, n_tokens, lr, weight_decay, pad_token_id, dropout)
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
        s2_y_pred = self.s2_model(lc_x_enc, lc_y_pred_enc) # (n_bptt, 1, bs_s2, n_tokens)
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
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(
                    optimizer, warmup_epochs=2000, max_epochs=100000, warmup_start_lr=0.0, eta_min=0.1*self.lr, last_epoch=-1), # torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100000, eta_min=0.1*self.lr),
                "interval": "step",
                "frequency": 1,
            },
        }



