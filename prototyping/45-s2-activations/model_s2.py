import torch, torch.nn as nn, torch.utils.data as data, torch.nn.functional as F
import lightning as L
import math
import torchmetrics
import random
import modelbasics as mb
import pl_bolts


class S1Model(nn.Module):
    def __init__(self, d_model, n_enc_heads, n_enc_layers, n_tokens, weight_decay, pad_token_id, dropout):
        super().__init__()

        self.d_model = d_model
        self.n_enc_heads = n_enc_heads
        self.n_enc_layers = n_enc_layers
        self.n_tokens = n_tokens
        self.weight_decay = weight_decay
        self.pad_token_id = pad_token_id
        self.dropout = dropout

        self.pred_enc = nn.TransformerEncoder(mb.TransformerEncoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        self.linear = nn.Linear(d_model, n_tokens)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, x_enc: torch.Tensor, padding_mask: torch.Tensor=None) -> torch.Tensor:
        """
        input:
        - x_enc: (seq_len, bs, d_model)
        - padding_mask: (bs, seq_len)
        returns 
        - y_pred_enc: (seq_len, bs, d_model)
        - y_pred: (seq_len, bs, n_tokens)
        """

        # make prediction
        y_pred_enc = self.pred_enc(x_enc, src_key_padding_mask=padding_mask) # (seq_len, bs, d_model)
        y_pred = self.linear(y_pred_enc) # (seq_len, bs, n_tokens)

        return y_pred_enc, y_pred






class S2Model(nn.Module):
    """
    This module is for System 2

    Simple start: Accepts encoded game states from VAE+S1, and predicts the next action.
    """

    def __init__(self, n_bptt, d_model, n_enc_heads, n_enc_layers, n_tokens, weight_decay, pad_token_id, dropout):
        super().__init__()

        self.n_bptt = n_bptt

        self.d_model = d_model
        self.wd = weight_decay
        self.pad_token_id = pad_token_id
        self.dropout = dropout
        self.n_tokens = n_tokens

        # dummy_s2_starter = torch.rand_like(torch.zeros(d_model)) * math.sqrt(d_model)
        # self.dummy_s2_starter = torch.nn.Parameter(dummy_s2_starter) # (d_model,)

        # models for translating to and from BPTT format
        self.enc_x_s2 = nn.TransformerEncoder(mb.TransformerEncoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        # self.dec_s2_1 = nn.TransformerDecoder(mb.TransformerDecoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        # self.dec_s2_2 = nn.TransformerDecoder(mb.TransformerDecoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        # self.linear_proj = nn.Linear(d_model, d_model)

        self.dec_s2 = nn.TransformerDecoder(mb.TransformerDecoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        self.linear_s2 = nn.Linear(d_model, n_tokens)

        # # models for game thinking
        # # self.enc_bptt = nn.TransformerEncoder(mb.TransformerEncoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        # self.enc_step1 = nn.TransformerEncoder(mb.TransformerEncoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        # self.dec_bptt = nn.TransformerDecoder(mb.TransformerDecoderLayer(d_model=d_model, nhead=n_enc_heads*2, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear_s2.bias.data.zero_()
        self.linear_s2.weight.data.uniform_(-initrange, initrange)

    def forward(self, x_enc: torch.Tensor, s1_activations: torch.Tensor, s1_model) -> torch.Tensor:
        """
        x_enc: (1, bs*seq_len, d_model)
        s1_activations: (n_enc_layers, bs*seq_len, d_model)
        returns y_pred: (n_bptt, seq_len, bs, n_tokens)

        No padding_mask required because we're not doing any attention over the sequence.
        So cross-enteropy loss will handle ignoring the padding tokens.
        """
        _, bs_sl, d_model = x_enc.shape

        # cut down activations (bs might be smaller than max_bs)
        s1_activations = s1_activations[:, :bs_sl] # (n_enc_layers, bs*seq_len, d_model)

        # translate x_enc into s2 format
        s2_x_enc = self.enc_x_s2(x_enc) # (1, bs*seq_len, d_model)

        # push through s2 model
        s2_pred_dec = self.dec_s2(s2_x_enc, s1_activations) # (1, bs*seq_len, d_model)
        s2_pred = self.linear_s2(s2_pred_dec) # (1, bs*seq_len, n_tokens)

        # # prep output tensor
        # # y_pred = torch.zeros(self.n_bptt, seq_len, bs, self.n_tokens, device=x_enc.device) # (n_bptt, seq_len, bs, n_tokens)

        # for i in range(8):
        #     # PHASE 1/2: update state
        #     # make s1 query (frozen model)
        #     s1_query = s2_state[-1:] # (1, bs*seq_len, d_model)
        #     s1_resp = self.s1_model.pred_enc(s1_query) # (1, bs*seq_len, d_model)
        #     s1_acts = torch.cat(self.s1_model_activations, dim=0) # (n_layers, bs*seq_len, d_model)
        #     # make x decoder input (encode)
        #     s2_state_enc = torch.zeros(2, bs*seq_len, d_model, device=x_enc.device) # (2, bs*seq_len, d_model)
        #     s2_state_enc[0, :] = s2_state[0, :]
        #     s2_state_enc[1, :] = self.enc_x(s2_state[:1]) # (bs*seq_len, d_model)
        #     # push through decoder: update state
        #     s2_phase1 = self.dec_s2_1(s2_state_enc, s1_acts) # (2, bs*seq_len, d_model)
        #     # PHASE 2/2: glance at original x_enc and update state for next cycle
        #     s2_state = self.dec_s2_2(s2_phase1, x_enc) # (2, bs*seq_len, d_model)

        # # make output pred
        # y_pred = self.linear_out(s2_phase1[1:]) # (bs*seq_len, n_tokens)

        # # shape for output
        # y_pred = y_pred.reshape(seq_len, bs, -1).unsqueeze(0) # (1, seq_len, bs, n_tokens)
        # return y_pred # (n_bptt, seq_len, bs, n_tokens)

        return s2_pred


# Define entire system as a LightningModule
class S2Transformer(L.LightningModule):
    def __init__(self, vae_model, d_model, n_enc_heads, n_enc_layers, n_tokens, lr_s1, lr_s2, weight_decay, pad_token_id, predictive, dropout, n_bptt, max_bs, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.d_model = d_model
        self.lr_s1 = lr_s1
        self.lr_s2 = lr_s2
        self.wd = weight_decay
        self.pad_token_id = pad_token_id
        self.predictive = predictive
        self.metrics = {}
        self.dropout = dropout
        self.n_bptt = n_bptt
        self.n_tokens = n_tokens

        # freeze pre-trained vae
        self.vae = vae_model
        for param in self.vae.parameters():
            param.requires_grad = False

        self.s1_model = S1Model(d_model, n_enc_heads, n_enc_layers, n_tokens, weight_decay, pad_token_id, dropout)
        self.s2_model = S2Model(self.n_bptt, d_model, n_enc_heads, n_enc_layers, n_tokens, weight_decay, pad_token_id, dropout)

        # setup hooks to get activations from each layer of s1_model
        self.s1_model_hooks = []
        self.s1_model_activations = torch.zeros(n_enc_layers, max_bs*512, d_model, device=self.vae.device) # (n_enc_layers, bs*seq_len, d_model)
        def get_hook(i):
            def hook(module, input, output):
                bs_sl = output.shape[1]
                self.s1_model_activations[i, :bs_sl] = output[0] # (bs*seq_len, d_model)
            return hook
        for i in range(len(self.s1_model.pred_enc.layers)):
            self.s1_model_hooks.append(self.s1_model.pred_enc.layers[i].register_forward_hook(get_hook(i)))



        # metrics
        for mode in ['train', 'val']: # hack: metrics must be on self or Lightning doesn't handle their devices correctly
            setattr(self, f'{mode}_s1_acc', torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_tokens, ignore_index=pad_token_id))
            self.metrics[f'{mode}_s1_acc'] = getattr(self, f'{mode}_s1_acc')
            setattr(self, f'{mode}_s2_acc', torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_tokens, ignore_index=pad_token_id))
            self.metrics[f'{mode}_s2_acc'] = getattr(self, f'{mode}_s2_acc')


    def model_step(self, batch, batch_idx, mode='train'):
        s1_opt, s2_opt = self.optimizers()
        s1_lr_sched, s2_lr_sched = self.lr_schedulers()

        x, y = batch # (bs, seq_len)
        padding_mask = (x==self.pad_token_id) # (bs, seq_len)
        x, y = x.to(torch.long), y.to(torch.long) # do conversion on GPU (mem bottleneck)

        # encode to single token with pre-trained, frozen VAE
        x_emb, x_enc = self.vae.forward_enc(x, padding_mask=padding_mask) # (seq_len, bs, d_model)
        seq_len, bs, d_model = x_enc.shape

        # since each token is its own row, convert to seq_len=1 for hygiene
        # (this makes padding irrelevant, since cross-entropy loss ignores padding tokens,
        # but it's a good habit to keep the shape consistent)
        x_enc = x_enc.reshape(1, -1, d_model) # (1, bs*seq_len, d_model)
        #y = y.reshape(-1) # (bs*seq_len,)
        padding_mask = padding_mask.reshape(-1,1) # (bs*seq_len,1)

        # ===========================
        # Predict, optimize S1 model
        # ===========================
        s1_pred_enc, s1_pred = self.s1_model(x_enc) # (1, bs*seq_len, d_model), (1, bs*seq_len, n_tokens)

        s1_pred = s1_pred.reshape(seq_len, bs, -1) # (seq_len, bs, n_tokens)
        s1_pred = s1_pred.permute(1, 0, 2) # (bs, seq_len, n_tokens)

        s1_pred_ce = torch.permute(s1_pred, (0, 2, 1)) # (bs, n_tokens, seq_len)
        s1_loss = F.cross_entropy(s1_pred_ce, y, ignore_index=self.pad_token_id)

        # s1_pred_ce = s1_pred.squeeze(0) # (bs*seq_len, n_tokens)
        # s1_loss = F.cross_entropy(s1_pred_ce, y, ignore_index=self.pad_token_id) # (1,)

        # log stats
        _, s1_y_hat = torch.max(s1_pred_ce, dim=1) # (bs*seq_len,)
        s1_acc = self.metrics[f'{mode}_s1_acc'](s1_y_hat, y)
        self.log(f"{mode}_s1_acc", s1_acc)

        # optimize
        if mode == 'train':
            s1_opt.zero_grad()
            self.manual_backward(s1_loss)
            torch.nn.utils.clip_grad_norm_(self.s1_model.parameters(), 1.0)
            s1_opt.step()
            s1_lr_sched.step()

        # ===========================
        # Predict, optimize S2 model
        # ===========================

        s2_pred = self.s2_model(x_enc, self.s1_model_activations.detach(), self.s1_model) # (1, bs*seq_len, n_tokens)

        s2_pred = s2_pred.reshape(seq_len, bs, -1) # (seq_len, bs, n_tokens)
        s2_pred = s2_pred.permute(1, 0, 2) # (bs, seq_len, n_tokens)

        s2_pred_ce = torch.permute(s2_pred, (0, 2, 1)) # (bs, n_tokens, seq_len)
        s2_loss = F.cross_entropy(s2_pred_ce, y, ignore_index=self.pad_token_id)

        # log stats
        _, s2_y_hat = torch.max(s2_pred_ce, dim=1) # (bs*seq_len,)
        s2_acc = self.metrics[f'{mode}_s2_acc'](s2_y_hat, y)
        self.log(f"{mode}_s2_acc", s2_acc)

        # optimize
        if mode == 'train':
            s2_opt.zero_grad()
            self.manual_backward(s2_loss)
            torch.nn.utils.clip_grad_norm_(self.s2_model.parameters(), 1.0)
            s2_opt.step()
            s2_lr_sched.step()


        # # Predict using frozen S1
        # x_enc_long = x_enc.reshape(1, -1, d_model) # (1, bs*seq_len, d_model)
        # y_pred_enc_long = self.s1_model.pred_enc(x_enc_long) # (1, bs*seq_len, d_model)
        # y_pred = self.s1_model.pred_linear(y_pred_enc_long) # (1, bs*seq_len, n_tokens)
        # y_pred = y_pred.reshape(seq_len, bs, -1) # (seq_len, bs, n_tokens)
        # y_pred = y_pred.permute(1, 0, 2) # (bs, seq_len, n_tokens)

        # # Use calibration to focus on low-confidence predictions
        # y_prob = F.softmax(y_pred, dim=2) # (bs, seq_len, n_tokens)
        # y_maxprob, y_hat = torch.max(y_prob, dim=2) # (bs, seq_len)

        # # grab anything < 0.95
        # threshold_conf, threshold_trim = 0.95, 0.125
        # focus_mask = torch.logical_and(y_maxprob < threshold_conf, ~padding_mask) # (bs, seq_len)
        # n_focus = focus_mask.sum()

        # # setup s2 inputs: trim into fixed shape
        # bs_s2 = int(bs*seq_len*threshold_trim)
        # lc_x_enc = torch.zeros(bs_s2, d_model, device=x_enc.device) + self.pad_token_id # (bs_s2, d_model)
        # lc_y_pred_enc = torch.zeros(bs_s2, d_model, device=x_enc.device) + self.pad_token_id # (bs_s2, d_model)
        # lc_y_pred = torch.zeros(bs_s2, self.n_tokens, device=x_enc.device) + self.pad_token_id # (bs_s2, n_tokens)
        # lc_y = torch.zeros(bs_s2, dtype=torch.long, device=x_enc.device) + self.pad_token_id # (bs_s2,)

        # lc_x_enc[:] = x_enc.transpose(0,1)[focus_mask][:bs_s2] # (bs_s2, d_model)
        # lc_y_pred_enc[:] = y_pred_enc_long.reshape(bs,seq_len,-1)[focus_mask][:bs_s2] # (bs_s2, d_model)
        # lc_y_pred[:] = y_pred[focus_mask][:bs_s2] # (bs_s2, n_tokens)
        # lc_y[:] = y[focus_mask][:bs_s2] # (bs_s2,)
        # lc_mask = lc_y == self.pad_token_id # (bs_s2,)

        # # log stats for low-confidence subset
        # lc_y_prob = F.softmax(lc_y_pred[:bs_s2], dim=1)
        # lc_y_prob_maxprob, lc_y_pred = lc_y_prob.max(dim=1)
        # lc_y_prob_avg_maxprob = lc_y_prob_maxprob.mean()
        # subset_acc = (lc_y_pred == lc_y)[:n_focus].float().mean()
        # self.log(f"{mode}_s1_lowconf_acc", subset_acc)
        # self.log(f"{mode}_s1_lowconf_conf", lc_y_prob_avg_maxprob)
    
        # # reshape to fit s2 model
        # lc_x_enc = lc_x_enc.unsqueeze(0) # (1, bs_s2, d_model)
        # lc_y_pred_enc = lc_y_pred_enc.unsqueeze(0) # (1, bs_s2, d_model)
        # lc_y_pred = lc_y_pred.unsqueeze(0) # (1, bs_s2, n_tokens)
        # lc_y = lc_y.unsqueeze(1) # (bs_s2,1)
        # lc_mask = lc_mask.unsqueeze(1) # (bs_s2,1)

        # # Predict using S2
        # s2_y_pred = self.s2_model(lc_x_enc, lc_y_pred_enc) # (n_bptt, 1, bs_s2, n_tokens)
        # n_bptt, _, bs_s2, n_tokens = s2_y_pred.shape
        # s2_y_pred = s2_y_pred.permute(0,2,3,1) # (n_bptt, bs_s2, n_tokens, 1)

        # s2_ce_y = lc_y.repeat(self.n_bptt, 1) # (n_bptt*bs_s2, 1)
        # s2_ce_y_pred = s2_y_pred.reshape(-1, n_tokens, 1) # (n_bptt*bs_s2, n_tokens, 1)
        # loss = F.cross_entropy(s2_ce_y_pred, s2_ce_y, ignore_index=self.pad_token_id)
        # self.log(f"{mode}_s2_pred_loss", loss) 

        # # accuracy: compare first-vs-last BPTT step
        # s2_y_hat_allsteps = torch.argmax(s2_y_pred, dim=2) # (n_bptt,bs,seq_len)
        # for i in range(self.n_bptt):
        #     acc = self.metrics[f'{mode}_pred_acc_b{i:02}'](s2_y_hat_allsteps[i], lc_y)
        #     self.log(f"{mode}_pred_acc_b{i:02}", acc)
        #     if i > 0:
        #         self.log(f"{mode}_pred_acc_b{i:02}_delta", acc - prev_acc)
        #     elif i == 0:
        #         first_acc = acc
        #     prev_acc = acc
        # self.log(f"{mode}_pred_acc_full_delta", acc - first_acc)

        # # recombine low-conf with full set and report overall accuracy
        # s1_acc = torch.sum(torch.logical_and(y_hat == y, ~padding_mask)) / torch.sum(~padding_mask)
        # self.log(f"{mode}_s1_acc", s1_acc)

        # n_edits = min(n_focus, bs_s2)
        # edit_segment = y_pred[focus_mask] # (n_focus, n_tokens)
        # edit_segment[:bs_s2] = s2_y_pred[-1].squeeze()[:n_edits]
        # y_pred[focus_mask] = edit_segment

        # #y_pred[focus_mask][:bs_s2] = s2_y_pred[-1].squeeze() # (bs, seq_len, n_tokens)
        # cmb_y_prob = F.softmax(y_pred, dim=2) # (bs, seq_len, n_tokens)
        # cmb_y_prob_maxprob, cmb_y_pred = cmb_y_prob.max(dim=2) # (bs, seq_len)
        # combined_acc = torch.sum(torch.logical_and(cmb_y_pred == y, ~padding_mask)) / torch.sum(~padding_mask)
        # self.log(f"{mode}_s1s2_acc", combined_acc)
        # self.log(f"{mode}_s1s2_acc_gain", combined_acc - s1_acc)
        
        self.log_dict({"s1_loss": s1_loss, "s2_loss": s2_loss}, prog_bar=True)

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
        s1_opt = torch.optim.AdamW(self.s1_model.parameters(), lr=self.lr_s1, weight_decay=self.wd)
        s2_opt = torch.optim.AdamW(self.s2_model.parameters(), lr=self.lr_s2, weight_decay=self.wd)
        s1_lr_sched = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(
            s1_opt, warmup_epochs=2000, max_epochs=100000, warmup_start_lr=0.0, eta_min=0.1*self.lr_s1, last_epoch=-1)
        s2_lr_sched = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(
            s2_opt, warmup_epochs=2000, max_epochs=100000, warmup_start_lr=0.0, eta_min=0.1*self.lr_s2, last_epoch=-1)

        return [s1_opt, s2_opt], [s1_lr_sched, s2_lr_sched] 

