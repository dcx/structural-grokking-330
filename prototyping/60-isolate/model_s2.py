import torch, torch.nn as nn, torch.utils.data as data, torch.nn.functional as F
import lightning as L
import math
import torchmetrics
import random
import modelbasics as mb



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

        self.embedding = nn.Embedding(n_tokens, d_model, padding_idx=pad_token_id)

        # encode: long input -> single token
        self.enc_in = nn.TransformerEncoder(mb.TransformerEncoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        self.dec_in = nn.TransformerDecoder(mb.TransformerDecoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        dec_in_starter = torch.rand_like(torch.zeros(d_model)) * math.sqrt(d_model)
        self.dec_in_starter = torch.nn.Parameter(dec_in_starter) # (d_model,)

        # x vae: decode: single token -> long output
        self.dec_out = nn.TransformerDecoder(mb.TransformerDecoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        dec_out_starter = torch.rand_like(torch.zeros(d_model)) * math.sqrt(d_model)
        self.dec_out_starter = torch.nn.Parameter(dec_out_starter) # (d_model,)
        self.dec_out_linear = nn.Linear(d_model, n_tokens)

        # s1 model
        self.s1_enc = nn.TransformerEncoder(mb.TransformerEncoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        self.s1_linear = nn.Linear(d_model, n_tokens)

        # extract + cleancut
        self.extract_enc = nn.TransformerEncoder(mb.TransformerEncoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        self.cleancut_enc = nn.TransformerEncoder(mb.TransformerEncoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)

        # recombine
        self.combine_dec = nn.TransformerDecoder(mb.TransformerDecoderLayer(d_model=d_model, nhead=n_enc_heads, activation=F.leaky_relu, dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)

        # metrics
        for mode in ['train', 'val']:
            setattr(self, f'{mode}_x_vae_acc', torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_tokens, ignore_index=pad_token_id))
            self.metrics[f'{mode}_x_vae_acc'] = getattr(self, f'{mode}_x_vae_acc')
            setattr(self, f'{mode}_x_s1_acc', torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_tokens, ignore_index=pad_token_id))
            self.metrics[f'{mode}_x_s1_acc'] = getattr(self, f'{mode}_x_s1_acc')
            setattr(self, f'{mode}_xt_vae_acc', torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_tokens, ignore_index=pad_token_id))
            self.metrics[f'{mode}_xt_vae_acc'] = getattr(self, f'{mode}_xt_vae_acc')
            setattr(self, f'{mode}_xt_s1_acc', torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_tokens, ignore_index=pad_token_id))
            self.metrics[f'{mode}_xt_s1_acc'] = getattr(self, f'{mode}_xt_s1_acc')
            setattr(self, f'{mode}_xnext_vae_acc', torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_tokens, ignore_index=pad_token_id))
            self.metrics[f'{mode}_xnext_vae_acc'] = getattr(self, f'{mode}_xnext_vae_acc')

            setattr(self, f'{mode}_xnext_vae_acc_rw', torchmetrics.classification.Accuracy(task="binary", num_classes=2))
            self.metrics[f'{mode}_xnext_vae_acc_rw'] = getattr(self, f'{mode}_xnext_vae_acc_rw')



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
        x, y, xnext, xsi, ysi = batch # (bs, seq_len)
        padding_mask = (x==self.pad_token_id) # (bs, seq_len)
        padding_mask_xt = (xnext==self.pad_token_id) # (bs, seq_len_xt)
        x, y = x.to(torch.long), y.to(torch.long) # do conversion on GPU (mem bottleneck)
        xnext, xsi, ysi = xnext.to(torch.long), xsi.to(torch.long), ysi.to(torch.long) # do conversion on GPU (mem bottleneck)


        # embed x
        x_emb = self.embedding(x.T) * math.sqrt(self.d_model) # (seq_len, bs, d_model)

        # encode long input into single token
        x_enc_seq = self.enc_in(x_emb, src_key_padding_mask=padding_mask) # (seq_len, bs, d_model)
        x_dec_in_starter = self.dec_in_starter.unsqueeze(0).unsqueeze(0).repeat(1, x_enc_seq.shape[1], 1) # (1, bs, d_model)
        x_enc = self.dec_in(x_dec_in_starter, x_enc_seq, memory_key_padding_mask=padding_mask) # (1, bs, d_model)

        # X VAE
        VAE_LOSS_UPWEIGHT_FACTOR = 2.5
        # x-vae: decode: single token into long output (teacher forcing)
        x_pred_in = torch.cat([
            self.dec_out_starter.unsqueeze(0).unsqueeze(0).repeat(1, x_enc_seq.shape[1], 1), # (1, bs, d_model)
            x_emb[:-1] # (seq_len-1, bs, d_model),
        ], dim=0) # (seq_len, bs, d_model)
        x_pred_padding_mask = torch.cat([
            torch.zeros(x_enc_seq.shape[1], 1, dtype=torch.bool, device=x_enc_seq.device), # (bs, 1)
            padding_mask[:,:-1] # (bs, seq_len-1)
        ], dim=1) # (bs, seq_len)
        cmask = torch.nn.Transformer.generate_square_subsequent_mask(sz=x_emb.shape[0]).to(x.device) # (seq_len, seq_len)
        x_dec_seq = self.dec_out(x_pred_in, x_enc, tgt_key_padding_mask=x_pred_padding_mask, tgt_mask=cmask) # (seq_len, bs, d_model)
        x_dec_seq = x_dec_seq.permute(1,0,2) # (bs, seq_len, d_model)
        x_pred_out = self.dec_out_linear(x_dec_seq) # (bs, seq_len, n_tokens)
        # loss,acc: x-vae
        x_pred_out_ce = x_pred_out.permute(0,2,1) # (bs, n_tokens, seq_len)
        loss = VAE_LOSS_UPWEIGHT_FACTOR*F.cross_entropy(x_pred_out_ce, x, ignore_index=self.pad_token_id)
        self.log(f"{mode}_01_x_vae_loss", loss, prog_bar=True)
        x_hat = torch.argmax(x_pred_out, dim=2) # (bs, seq_len)
        acc = self.metrics[f'{mode}_x_vae_acc'](x_hat, x)
        self.log(f"{mode}_01_x_vae_acc", acc, prog_bar=True)

        # X-S1
        # x-s1: predict result over full x
        s1_enc = self.s1_enc(x_enc) # (1, bs, d_model)
        s1_pred = self.s1_linear(s1_enc) # (1, bs, n_tokens)
        # loss,acc: x-s1
        s1_pred_ce = s1_pred.permute(1,2,0) # (bs, n_tokens, 1)
        loss_s1 = F.cross_entropy(s1_pred_ce, y.unsqueeze(1), ignore_index=self.pad_token_id)
        self.log(f"{mode}_02_x_s1_loss", loss_s1, prog_bar=False)
        loss += loss_s1
        s1_hat = torch.argmax(s1_pred, dim=2).squeeze(0) # (bs,)
        acc = self.metrics[f'{mode}_x_s1_acc'](s1_hat, y)
        self.log(f"{mode}_02_x_s1_acc", acc, prog_bar=False)


        # XT-VAE
        # extract: get view of x_enc where current action is shadowed - e.g.: (*3(+23)) -> (*3????)
        xt_delta_enc = self.extract_enc(x_enc) # (1, bs, d_model)
        # clean cut: convert current action to its own view - e.g.: ???(+23)? -> (+23)____
        xt_enc = self.cleancut_enc(x_enc - xt_delta_enc) # (1, bs, d_model) 

        # xt-decode: teacher-force extract
        xt_emb = self.embedding(xsi.T) * math.sqrt(self.d_model) # (seq_len_xt, bs, d_model)
        xt_pred_in = torch.cat([
            self.dec_out_starter.unsqueeze(0).unsqueeze(0).repeat(1, x_enc_seq.shape[1], 1), # (1, bs, d_model)
            xt_emb[:-1] # (seq_len_xt-1, bs, d_model),
        ], dim=0) # (seq_len_xt, bs, d_model)
        xt_pred_padding_mask = torch.cat([
            torch.zeros(x_enc_seq.shape[1], 1, dtype=torch.bool, device=x_enc_seq.device), # (bs, 1)
            padding_mask_xt[:,:-1] # (bs, seq_len_xt-1)
        ], dim=1) # (bs, seq_len)
        cmask_xt = torch.nn.Transformer.generate_square_subsequent_mask(sz=xt_emb.shape[0]).to(x.device) # (seq_len, seq_len)
        xt_dec_seq = self.dec_out(xt_pred_in, xt_enc, tgt_key_padding_mask=xt_pred_padding_mask, tgt_mask=cmask_xt) # (seq_len_xt, bs, d_model)
        xt_dec_seq = xt_dec_seq.permute(1,0,2) # (bs, seq_len, d_model)
        xt_pred_out = self.dec_out_linear(xt_dec_seq) # (bs, seq_len, n_tokens)

        # loss,acc: xt-vae
        xt_pred_out_ce = xt_pred_out.permute(0,2,1) # (bs, n_tokens, seq_len)
        loss_xt = F.cross_entropy(xt_pred_out_ce, xnext, ignore_index=self.pad_token_id)
        self.log(f"{mode}_03_xt_vae_loss", loss_xt, prog_bar=False)
        loss += loss_xt
        xt_hat = torch.argmax(xt_pred_out, dim=2) # (bs, seq_len)
        acc = self.metrics[f'{mode}_xt_vae_acc'](xt_hat, xnext)
        self.log(f"{mode}_03_xt_vae_acc", acc, prog_bar=False)

        # XT-S1
        # xt-s1: predict result over extract
        s1_enc_xt = self.s1_enc(xt_enc) # (1, bs, d_model)
        s1_pred_xt = self.s1_linear(s1_enc_xt) # (1, bs, n_tokens)
        # loss,acc: xt-s1
        s1_pred_xt_ce = s1_pred_xt.permute(1,2,0) # (bs, n_tokens, 1)
        loss_s1_xt = F.cross_entropy(s1_pred_xt_ce, ysi.unsqueeze(1), ignore_index=self.pad_token_id)
        self.log(f"{mode}_04_xt_s1_loss", loss_s1_xt, prog_bar=False)
        loss += loss_s1_xt
        s1_hat_xt = torch.argmax(s1_pred_xt, dim=2).squeeze(0) # (bs,)
        acc = self.metrics[f'{mode}_xt_s1_acc'](s1_hat_xt, ysi)
        self.log(f"{mode}_04_xt_s1_acc", acc, prog_bar=False)


        # COMBINE: delta + pred -> next state
        ysi_enc = self.embedding(ysi.unsqueeze(0)) * math.sqrt(self.d_model) # (1, bs, d_model)
        xnext_enc = self.combine_dec(xt_delta_enc, torch.cat([ysi_enc,x_enc],dim=0)) # (1, bs, d_model)
        # teacher-force decode vs xnext
        padding_mask_xnext = (xnext==self.pad_token_id) # (bs, seq_len_xnext)
        xnext_emb = self.embedding(xnext.T) * math.sqrt(self.d_model) # (seq_len_xnext, bs, d_model)
        xnext_pred_in = torch.cat([
            self.dec_out_starter.unsqueeze(0).unsqueeze(0).repeat(1, x_enc_seq.shape[1], 1), # (1, bs, d_model)
            xnext_emb[:-1] # (seq_len_xnext-1, bs, d_model),
        ], dim=0)
        xnext_pred_padding_mask = torch.cat([
            torch.zeros(x_enc_seq.shape[1], 1, dtype=torch.bool, device=x_enc_seq.device), # (bs, 1)
            padding_mask_xnext[:,:-1] # (bs, seq_len_xnext-1)
        ], dim=1)
        cmask_xnext = torch.nn.Transformer.generate_square_subsequent_mask(sz=xnext_emb.shape[0]).to(x.device) # (seq_len_xnext, seq_len_xnext)
        xnext_dec_seq = self.dec_out(xnext_pred_in, xnext_enc, tgt_key_padding_mask=xnext_pred_padding_mask, tgt_mask=cmask_xnext) # (seq_len_xnext, bs, d_model)
        xnext_dec_seq = xnext_dec_seq.permute(1,0,2) # (bs, seq_len_xnext, d_model)
        xnext_pred_out = self.dec_out_linear(xnext_dec_seq) # (bs, seq_len_xnext, n_tokens)

        # loss,acc: xnext-vae
        xnext_pred_out_ce = xnext_pred_out.permute(0,2,1) # (bs, n_tokens, seq_len_xnext)
        loss_xnext = F.cross_entropy(xnext_pred_out_ce, xnext, ignore_index=self.pad_token_id)
        self.log(f"{mode}_05_xnext_vae_loss", loss_xnext, prog_bar=False)
        loss += loss_xnext
        xnext_hat = torch.argmax(xnext_pred_out, dim=2) # (bs, seq_len_xnext)
        acc = self.metrics[f'{mode}_xnext_vae_acc'](xnext_hat, xnext)
        self.log(f"{mode}_05_xnext_vae_acc", acc, prog_bar=False)

        # rowwise accuracy
        xnext_hat_rw = (xnext_hat == xnext).all(dim=1)
        acc = self.metrics[f'{mode}_xnext_vae_acc_rw'](xnext_hat_rw, torch.ones_like(xnext_hat_rw))
        self.log(f"{mode}_05_xnext_vae_acc_rw", acc, prog_bar=False)
        
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




