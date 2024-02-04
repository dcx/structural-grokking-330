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

        # setup hooks to get activations from each layer of s1_model
        self.hooks = []
        self.activations = [None] * n_enc_layers
        def get_hook(i):
            def hook(module, input, output):
                self.activations[i] = output.reshape(1, -1, d_model) # (1, bs*seq_len, d_model)
            return hook
        for i in range(len(self.pred_enc.layers)):
            self.hooks.append(self.pred_enc.layers[i].register_forward_hook(get_hook(i)))


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

    def __init__(self, n_bptt, d_model, n_enc_heads, n_enc_layers, n_tokens, weight_decay, pad_token_id, dropout, n_s2_loops):
        super().__init__()

        self.n_bptt = n_bptt

        self.d_model = d_model
        self.wd = weight_decay
        self.pad_token_id = pad_token_id
        self.dropout = dropout
        self.n_tokens = n_tokens
        self.n_s2_loops = n_s2_loops

        dummy_s2_starter = torch.rand_like(torch.zeros(d_model)) * math.sqrt(d_model)
        self.dummy_s2_starter = torch.nn.Parameter(dummy_s2_starter) # (d_model,)

        # models for translating to and from BPTT format
        self.enc_x_s2 = nn.TransformerEncoder(mb.TransformerEncoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        self.dec_s2_1 = nn.TransformerDecoder(mb.TransformerDecoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        self.dec_s2_2 = nn.TransformerDecoder(mb.TransformerDecoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        # self.linear_proj = nn.Linear(d_model, d_model)

        # self.dec_s2 = nn.TransformerDecoder(mb.TransformerDecoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
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

    def forward(self, x_enc: torch.Tensor, s1_model) -> torch.Tensor:
        """
        x_enc: (1, bs*seq_len, d_model)
        s1_activations: (n_enc_layers, bs*seq_len, d_model)
        returns y_pred: (n_bptt, seq_len, bs, n_tokens)

        No padding_mask required because we're not doing any attention over the sequence.
        So cross-enteropy loss will handle ignoring the padding tokens.
        """
        _, bs_sl, d_model = x_enc.shape

        # prep s2 state
        cur_x_enc = x_enc # (1, bs_sl, d_model)
        s2_state = self.dummy_s2_starter.clone().repeat(1, bs_sl, 1) # (1, bs_sl, d_model)

        for i in range(self.n_s2_loops):
            # PHASE 1/2: make s1 query and update state
            s2_enc = self.enc_x_s2(cur_x_enc) # (1, bs_sl, d_model)
            s1_fwd = s1_model.pred_enc(cur_x_enc).detach() # (1, bs_sl, d_model)
            if i == 0:
                s1_save = s1_fwd
            s1_activations = torch.cat(s1_model.activations, dim=0).detach() # (n_layers, bs_sl, d_model)
            s2_phase_1 = self.dec_s2_1(torch.cat((s2_state, s2_enc), dim=0), s1_activations) # (2, bs_sl, d_model)

            # PHASE 2/2: glance at original x_enc, make S1 query
            s2_update = self.dec_s2_2(s2_phase_1, cur_x_enc) # (2, bs_sl, d_model)
            s2_state = s2_update[:1] # (1, bs_sl, d_model)
            cur_x_enc = s2_update[1:] # (1, bs_sl, d_model)

        s2_pred = self.linear_s2(s1_save + s2_phase_1[-1:]) # (1, bs_sl, n_tokens)

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
    def __init__(self, vae_model, d_model, n_enc_heads, n_enc_layers, n_tokens, lr_s1, lr_s2, weight_decay, pad_token_id, predictive, dropout, n_bptt, max_bs, n_s2_loops, **kwargs):
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
        self.s2_model = S2Model(self.n_bptt, d_model, n_enc_heads, n_enc_layers, n_tokens, weight_decay, pad_token_id, dropout, n_s2_loops)

        # metrics
        for mode in ['train', 'val']: # hack: metrics must be on self or Lightning doesn't handle their devices correctly
            setattr(self, f'{mode}_s1_acc', torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_tokens, ignore_index=pad_token_id))
            self.metrics[f'{mode}_s1_acc'] = getattr(self, f'{mode}_s1_acc')
            setattr(self, f'{mode}_s2_lc_acc', torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_tokens, ignore_index=pad_token_id))
            self.metrics[f'{mode}_s2_lc_acc'] = getattr(self, f'{mode}_s2_lc_acc')
            setattr(self, f'{mode}_cmb_acc', torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_tokens, ignore_index=pad_token_id))
            self.metrics[f'{mode}_cmb_acc'] = getattr(self, f'{mode}_cmb_acc')


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
        # Pull out low-confidence predictions
        # ===========================

        s2_fraction = 0.25
        n_lc = int(bs*seq_len*s2_fraction)
        s1_prob = F.softmax(s1_pred_enc, dim=2) # (1, bs*seq_len, n_tokens)
        s1_maxprob, _ = torch.max(s1_prob, dim=2) # (1, bs*seq_len)
        s1_maxprob_nonpad = s1_maxprob * ~padding_mask.squeeze(1) # (1, bs*seq_len)

        s1_max_idxs = torch.argsort(s1_maxprob_nonpad, descending=True, dim=1)[:,:n_lc] # (n_lc,)
        lc_x_enc = x_enc[:,s1_max_idxs[0]] # (1, n_lc, d_model)
        lc_y = y.T.reshape(-1)[s1_max_idxs[0]] # (n_lc,)


        # ===========================
        # Predict, optimize S2 model
        # ===========================

        s2_lc_pred = self.s2_model(lc_x_enc, self.s1_model) # (1, n_lc, n_tokens)

        #s2_pred = s2_pred.reshape(seq_len, bs, -1) # (seq_len, bs, n_tokens)
        #s2_pred = s2_pred.permute(1, 0, 2) # (bs, seq_len, n_tokens)

        s2_lc_pred_ce = s2_lc_pred.squeeze(0) # (n_lc, n_tokens)
        s2_loss = F.cross_entropy(s2_lc_pred_ce, lc_y, ignore_index=self.pad_token_id)

        # log stats
        _, s2_lc_y_hat = torch.max(s2_lc_pred_ce, dim=1) # (n_lc,)
        s2_lc_acc = self.metrics[f'{mode}_s2_lc_acc'](s2_lc_y_hat, lc_y)
        self.log(f"{mode}_s2_lc_acc", s2_lc_acc)

        # optimize
        if mode == 'train':
            s2_opt.zero_grad()
            self.manual_backward(s2_loss)
            torch.nn.utils.clip_grad_norm_(self.s2_model.parameters(), 1.0)
            s2_opt.step()
            s2_lr_sched.step()

        # recombine with S1 preds to report overall accuracy
        cmb_y_hat = s1_y_hat.T.reshape(-1) # (bs*seq_len,)
        cmb_y_hat[s1_max_idxs[0]] = s2_lc_y_hat # (bs*seq_len,)
        cmb_acc = self.metrics[f'{mode}_cmb_acc'](cmb_y_hat, y.T.reshape(-1))
        self.log(f"{mode}_cmb_acc", cmb_acc)
        
        self.log(f"{mode}_loss_delta", s1_loss - s2_loss)
        self.log(f"{mode}_acc_delta", cmb_acc - s1_acc)

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

