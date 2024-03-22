from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import torch, torch.nn as nn, torch.utils.data as data, torch.nn.functional as F
import lightning as L
import math
import torchmetrics
import random
import modelbasics as mb
import pl_bolts
import dataset


# Define entire system as a LightningModule
class S2Transformer(L.LightningModule):
    def __init__(self, d_model, n_enc_heads, n_enc_layers, n_tokens, lr, weight_decay, pad_token_id, predictive, dropout, n_bptt, **kwargs): # dl_train, dl_val # for LR finder
        super().__init__()
        self.save_hyperparameters()

        # for learning rate finder
        # self.dl_train = dl_train
        # self.dl_val = dl_val

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

        # contains: predict if a subseq came from a token
        self.contains_dec = nn.TransformerDecoder(mb.TransformerDecoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        self.contains_linear = nn.Linear(d_model, 2)



        # # extract + cleancut
        # self.extract_enc = nn.TransformerEncoder(mb.TransformerEncoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        # self.cleancut_dec = nn.TransformerDecoder(mb.TransformerDecoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)

        # # recombine
        # self.combine_dec = nn.TransformerDecoder(mb.TransformerDecoderLayer(d_model=d_model, nhead=n_enc_heads, activation=F.leaky_relu, dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)

        # metrics
        for mode in ['train', 'val']:
            setattr(self, f'{mode}_x_vae_acc', torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_tokens, ignore_index=pad_token_id))
            self.metrics[f'{mode}_x_vae_acc'] = getattr(self, f'{mode}_x_vae_acc')
            setattr(self, f'{mode}_x_vae_acc_rw', torchmetrics.classification.Accuracy(task="binary", num_classes=2))
            self.metrics[f'{mode}_x_vae_acc_rw'] = getattr(self, f'{mode}_x_vae_acc')

            setattr(self, f'{mode}_x_s1_acc', torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_tokens, ignore_index=pad_token_id))
            self.metrics[f'{mode}_x_s1_acc'] = getattr(self, f'{mode}_x_s1_acc')

            setattr(self, f'{mode}_contains_acc', torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_tokens, ignore_index=pad_token_id))
            self.metrics[f'{mode}_contains_acc'] = getattr(self, f'{mode}_contains_acc')
            
            # setattr(self, f'{mode}_xt_vae_acc', torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_tokens, ignore_index=pad_token_id))
            # self.metrics[f'{mode}_xt_vae_acc'] = getattr(self, f'{mode}_xt_vae_acc')
            # setattr(self, f'{mode}_xt_s1_acc', torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_tokens, ignore_index=pad_token_id))
            # self.metrics[f'{mode}_xt_s1_acc'] = getattr(self, f'{mode}_xt_s1_acc')
            # setattr(self, f'{mode}_xnext_vae_acc', torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_tokens, ignore_index=pad_token_id))
            # self.metrics[f'{mode}_xnext_vae_acc'] = getattr(self, f'{mode}_xnext_vae_acc')

            # setattr(self, f'{mode}_xnext_vae_acc_rw', torchmetrics.classification.Accuracy(task="binary", num_classes=2))
            # self.metrics[f'{mode}_xnext_vae_acc_rw'] = getattr(self, f'{mode}_xnext_vae_acc_rw')



        self.init_weights()

        # # s2 model
        # self.s2_model = S2Model(self.n_bptt, d_model, n_enc_heads, n_enc_layers, n_tokens, lr, weight_decay, pad_token_id, dropout)
        # self.init_weights()

        # # metrics
        # for mode in ['train', 'val']: # hack: metrics must be on self or Lightning doesn't handle their devices correctly
        #     for i in range(self.n_bptt):
        #         setattr(self, f'{mode}_pred_acc_b{i:02}', torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_tokens, ignore_index=pad_token_id))
        #         self.metrics[f'{mode}_pred_acc_b{i:02}'] = getattr(self, f'{mode}_pred_acc_b{i:02}')

    # # for LR finder            
    # def train_dataloader(self):
    #     return self.dl_train
    # def val_dataloader(self):
    #     return self.dl_val


    def init_weights(self) -> None:
        initrange = 0.1
        self.dec_out_linear.bias.data.zero_()
        self.dec_out_linear.weight.data.uniform_(-initrange, initrange)
        self.contains_linear.bias.data.zero_()
        self.contains_linear.weight.data.uniform_(-initrange, initrange)


    def model_step(self, batch, batch_idx, mode='train'):
        height, length, x, y = batch # (bs, seq_len)
        padding_mask = (x==self.pad_token_id) # (bs, seq_len)
        x, y = x.to(torch.long), y.to(torch.long) # do conversion on GPU (mem bottleneck)
        bs, seq_len = x.shape

        # robustness training: create a second x where tokens are randomly corrupted
        # (it's used in the s1 model below)
        x_corrupt = x.clone() # (bs, seq_len)
        x_random = torch.randint(0, dataset.max_sym_token_id, x.shape, device=x.device)
        corrupt_percent = (torch.rand(x.shape[0], device=x.device) * 0.3).unsqueeze(1) # (bs,1) # corrupt 0-30% of the tokens
        x_should_corrupt = torch.logical_and(torch.rand(x.shape, device=x.device) < corrupt_percent, torch.logical_not(padding_mask))
        # always corrupt two tokens within 0-length for each row
        # (e.g. if length is 4 = indexes 0,1,2,3, so we multiply by 4 and floor)
        x_must_corrupt_idx1 = torch.floor(torch.rand(x.shape[0], device=x.device) * length).to(torch.long) # (bs,)
        x_must_corrupt_idx2 = torch.floor(torch.rand(x.shape[0], device=x.device) * length).to(torch.long) # (bs,)
        x_should_corrupt[torch.arange(x.shape[0]), x_must_corrupt_idx1] = True
        x_should_corrupt[torch.arange(x.shape[0]), x_must_corrupt_idx2] = True
        x_corrupt[x_should_corrupt] = x_random[x_should_corrupt]

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
            x_emb # (seq_len, bs, d_model),
        ], dim=0) # (seq_len+1, bs, d_model)
        x_pred_padding_mask = torch.cat([
            torch.zeros(x_enc_seq.shape[1], 1, dtype=torch.bool, device=x_enc_seq.device), # (bs, 1)
            padding_mask # (bs, seq_len)
        ], dim=1) # (bs, seq_len+1)
        cmask = torch.nn.Transformer.generate_square_subsequent_mask(sz=x_emb.shape[0]+1).to(x.device) # (seq_len+1, seq_len+1)
        x_dec_seq = self.dec_out(x_pred_in, x_enc, tgt_key_padding_mask=x_pred_padding_mask, tgt_mask=cmask) # (seq_len+1, bs, d_model)
        x_dec_seq = x_dec_seq.permute(1,0,2) # (bs, seq_len+1, d_model)
        x_pred_out = self.dec_out_linear(x_dec_seq) # (bs, seq_len+1, n_tokens)

        # loss,acc: x-vae
        x_pred_out_ce = x_pred_out.permute(0,2,1) # (bs, n_tokens, seq_len+1)
        # mark EOS token at length
        x_with_eos = torch.zeros(x.shape[0], x.shape[1]+1, dtype=torch.long, device=x.device) + self.pad_token_id
        x_with_eos[:,:-1] = x
        x_with_eos[torch.arange(x.shape[0]), length] = dataset.eos_token_id # (bs, seq_len+1)

        loss = VAE_LOSS_UPWEIGHT_FACTOR*F.cross_entropy(x_pred_out_ce, x_with_eos, ignore_index=self.pad_token_id)
        self.log(f"{mode}_01_x_vae_loss", loss, prog_bar=True)
        x_hat = torch.argmax(x_pred_out, dim=2) # (bs, seq_len+1)
        acc = self.metrics[f'{mode}_x_vae_acc'](x_hat, x_with_eos)
        self.log(f"{mode}_01_x_vae_acc", acc, prog_bar=True)
        # rowwise
        x_hat_rw = torch.logical_or(x_hat == x_with_eos, x_with_eos == self.pad_token_id).all(dim=1)
        acc = self.metrics[f'{mode}_x_vae_acc_rw'](x_hat_rw, torch.ones_like(x_hat_rw))
        self.log(f"{mode}_01_x_vae_acc_rw", acc, prog_bar=True)

        # X-S1

        # robustness training: for each batch, only use items with height<=3,
        # corrupt the items with height>3 and use them to learn to pred bad_token

        # filter: only use items with height <= 3
        y_filtered = y[height <= 3] # (bs_filt,)
        x_enc_filtered = x_enc[:,height <= 3] # (1, bs_filt, d_model)
        pad_filt = padding_mask[height <= 3]

        # grab corrupted versions of same items        
        x_emb_corrupt = self.embedding(x_corrupt[height <= 3].T) * math.sqrt(self.d_model) # (seq_len, bs, d_model)
        x_enc_seq_corrupt = self.enc_in(x_emb_corrupt, src_key_padding_mask=pad_filt) # (seq_len, bs, d_model)
        x_dec_in_starter_corrupt = self.dec_in_starter.unsqueeze(0).unsqueeze(0).repeat(1, x_enc_seq_corrupt.shape[1], 1) # (1, bs, d_model)
        x_enc_corrupt = self.dec_in(x_dec_in_starter_corrupt, x_enc_seq_corrupt, memory_key_padding_mask=pad_filt) # (1, bs, d_model)
        y_corrupt = torch.zeros_like(y_filtered) # (bs_filt,)
        y_corrupt[:] = dataset.bad_token_id

        x_enc_combined = torch.cat([x_enc_filtered, x_enc_corrupt], dim=1) # (1, 2*bs_filt, d_model)
        y_combined = torch.cat([y_filtered, y_corrupt], dim=0) # (2*bs_filt,)

        # x-s1: predict result over full x
        s1_enc = self.s1_enc(x_enc_combined) # (1, 2*bs_filt, d_model)
        s1_pred = self.s1_linear(s1_enc) # (1, 2*bs_filt, n_tokens)
        # loss,acc: x-s1
        s1_pred_ce = s1_pred.permute(1,2,0) # (2*bs_filt, n_tokens, 1)
        loss_s1 = F.cross_entropy(s1_pred_ce, y_combined.unsqueeze(1), ignore_index=self.pad_token_id)
        self.log(f"{mode}_02_x_s1_loss", loss_s1, prog_bar=False)
        loss += loss_s1
        s1_hat = torch.argmax(s1_pred, dim=2).squeeze(0) # (2*bs_filt,)
        acc = self.metrics[f'{mode}_x_s1_acc'](s1_hat, y_combined)
        self.log(f"{mode}_02_x_s1_acc", acc, prog_bar=False)


        # CONTAINS

        x_padded = torch.cat([x, torch.zeros(bs, 1, dtype=torch.long, device=x.device) + self.pad_token_id], dim=1) # (bs, seq_len+1)
        pad_idx = x.shape[1] # if seq_len is 4, guaranteed pad_idx in x_padded is 4

        # pull out random subsequence and encode
        length_subseq_rand = torch.floor(torch.rand(length.shape, device=length.device) * length).to(torch.long) + 1 # (bs,) range [1-length]
        length_subseq_max_start = length - length_subseq_rand # (bs,)
        length_subseq_start = torch.floor(torch.rand(length.shape, device=length.device) * length_subseq_max_start).to(torch.long) # (bs,)
        length_subseq_end = length_subseq_start + length_subseq_rand # (bs,)
        length_subseq_max = torch.max(length_subseq_rand) # scalar

        # (clever indexing to save a for loop)
        # create range tensor matching the shape needed for broadcasting
        range_tensor = torch.arange(length_subseq_max, device=length.device).unsqueeze(0) # 2D for broadcasting (1, length_subseq_max)
        # make segment indices based on segment starts and range tensor
        start_indices = length_subseq_start.unsqueeze(1) # (bs, 1) 
        end_indices = length_subseq_end.unsqueeze(1) # (bs, 1)
        segment_indices = start_indices + range_tensor # (bs, length_subseq_max)
        # replace indices past end with pad_idx
        segment_indices[segment_indices >= end_indices] = pad_idx

        x_ss = x_padded[torch.arange(bs).unsqueeze(1), segment_indices] # (bs, length_subseq_max)
        padding_mask_ss = (x_ss==self.pad_token_id) # (bs, length_subseq_max)
        x_emb_ss = self.embedding(x_ss.T) * math.sqrt(self.d_model) # (length_subseq_max, bs, d_model)
        x_enc_seq_ss = self.enc_in(x_emb_ss, src_key_padding_mask=padding_mask_ss) # (length_subseq_max, bs, d_model)
        x_dec_in_starter_ss = self.dec_in_starter.unsqueeze(0).unsqueeze(0).repeat(1, bs, 1) # (1, bs, d_model)
        x_enc_ss_pos = self.dec_in(x_dec_in_starter_ss, x_enc_seq_ss, memory_key_padding_mask=padding_mask_ss) # (1, bs, d_model)

        # negative case 1: roll along batch dim 
        x_enc_ss_neg = torch.roll(x_enc_ss_pos, 1, 1) # (1, bs, d_model)
        # TODO: negative case 2: corrupt encodings from above

        # predict if subseq is in x 
        contains_enc = self.contains_dec(
            x_enc.repeat(1,2,1), # (1, 2*bs, d_model)
            torch.cat([x_enc_ss_pos, x_enc_ss_neg], dim=1), # (1, 2*bs, d_model)
        ) # (1, 2*bs, d_model)
        contains_y = torch.cat([torch.ones(bs, dtype=torch.long, device=x.device), torch.zeros(bs, dtype=torch.long, device=x.device)], dim=0) # (2*bs,)

        contains_pred = self.contains_linear(contains_enc) # (1, 2*bs, 2)
        # loss,acc: contains
        contains_pred_ce = contains_pred.permute(1,2,0) # (2*bs, 2, 1)
        loss_contains = F.cross_entropy(contains_pred_ce, contains_y.unsqueeze(1), ignore_index=self.pad_token_id)
        self.log(f"{mode}_03_contains_loss", loss_contains, prog_bar=False)
        loss += loss_contains
        contains_hat = torch.argmax(contains_pred, dim=2).squeeze(0) # (bs,)
        acc = self.metrics[f'{mode}_contains_acc'](contains_hat, contains_y)
        self.log(f"{mode}_03_contains_acc", acc, prog_bar=False)



        # # XT-VAE
        # # pull out the specific subaction (e.g. (*3(+23)) -> (+23))
        # xt_enc = self.extract_enc(x_enc) # (1, bs, d_model)
        # # delta: make a view of the state ready to be combined with the next action (e.g. (*3(+23)) -> (*3????))
        # # xt_delta_enc = self.cleancut_dec(x_enc, xt_enc) # (1, bs, d_model) 

        # # xt-decode: teacher-force extract
        # xt_emb = self.embedding(xsi.T) * math.sqrt(self.d_model) # (seq_len_xt, bs, d_model)
        # xt_pred_in = torch.cat([
        #     self.dec_out_starter.unsqueeze(0).unsqueeze(0).repeat(1, x_enc_seq.shape[1], 1), # (1, bs, d_model)
        #     xt_emb[:-1] # (seq_len_xt-1, bs, d_model),
        # ], dim=0) # (seq_len_xt, bs, d_model)
        # xt_pred_padding_mask = torch.cat([
        #     torch.zeros(x_enc_seq.shape[1], 1, dtype=torch.bool, device=x_enc_seq.device), # (bs, 1)
        #     padding_mask_xt[:,:-1] # (bs, seq_len_xt-1)
        # ], dim=1) # (bs, seq_len)
        # cmask_xt = torch.nn.Transformer.generate_square_subsequent_mask(sz=xt_emb.shape[0]).to(x.device) # (seq_len, seq_len)
        # xt_dec_seq = self.dec_out(xt_pred_in, xt_enc, tgt_key_padding_mask=xt_pred_padding_mask, tgt_mask=cmask_xt) # (seq_len_xt, bs, d_model)
        # xt_dec_seq = xt_dec_seq.permute(1,0,2) # (bs, seq_len, d_model)
        # xt_pred_out = self.dec_out_linear(xt_dec_seq) # (bs, seq_len, n_tokens)

        # # loss,acc: xt-vae
        # xt_pred_out_ce = xt_pred_out.permute(0,2,1) # (bs, n_tokens, seq_len)
        # loss_xt = F.cross_entropy(xt_pred_out_ce, xnext, ignore_index=self.pad_token_id)
        # self.log(f"{mode}_03_xt_vae_loss", loss_xt, prog_bar=False)
        # loss += loss_xt
        # xt_hat = torch.argmax(xt_pred_out, dim=2) # (bs, seq_len)
        # acc = self.metrics[f'{mode}_xt_vae_acc'](xt_hat, xnext)
        # self.log(f"{mode}_03_xt_vae_acc", acc, prog_bar=False)

        # # XT-S1
        # # xt-s1: predict result over extract
        # s1_enc_xt = self.s1_enc(xt_enc) # (1, bs, d_model)
        # s1_pred_xt = self.s1_linear(s1_enc_xt) # (1, bs, n_tokens)
        # # loss,acc: xt-s1
        # s1_pred_xt_ce = s1_pred_xt.permute(1,2,0) # (bs, n_tokens, 1)
        # loss_s1_xt = F.cross_entropy(s1_pred_xt_ce, ysi.unsqueeze(1), ignore_index=self.pad_token_id)
        # self.log(f"{mode}_04_xt_s1_loss", loss_s1_xt, prog_bar=False)
        # loss += loss_s1_xt
        # s1_hat_xt = torch.argmax(s1_pred_xt, dim=2).squeeze(0) # (bs,)
        # acc = self.metrics[f'{mode}_xt_s1_acc'](s1_hat_xt, ysi)
        # self.log(f"{mode}_04_xt_s1_acc", acc, prog_bar=False)


        # # COMBINE: delta + pred -> next state
        # ysi_enc = self.embedding(ysi.unsqueeze(0)) * math.sqrt(self.d_model) # (1, bs, d_model)
        # xnext_enc = self.combine_dec(x_enc, torch.cat([xt_enc, ysi_enc], dim=0)) # (1, bs, d_model)
        # # teacher-force decode vs xnext
        # padding_mask_xnext = (xnext==self.pad_token_id) # (bs, seq_len_xnext)
        # xnext_emb = self.embedding(xnext.T) * math.sqrt(self.d_model) # (seq_len_xnext, bs, d_model)
        # xnext_pred_in = torch.cat([
        #     self.dec_out_starter.unsqueeze(0).unsqueeze(0).repeat(1, x_enc_seq.shape[1], 1), # (1, bs, d_model)
        #     xnext_emb[:-1] # (seq_len_xnext-1, bs, d_model),
        # ], dim=0)
        # xnext_pred_padding_mask = torch.cat([
        #     torch.zeros(x_enc_seq.shape[1], 1, dtype=torch.bool, device=x_enc_seq.device), # (bs, 1)
        #     padding_mask_xnext[:,:-1] # (bs, seq_len_xnext-1)
        # ], dim=1)
        # cmask_xnext = torch.nn.Transformer.generate_square_subsequent_mask(sz=xnext_emb.shape[0]).to(x.device) # (seq_len_xnext, seq_len_xnext)
        # xnext_dec_seq = self.dec_out(xnext_pred_in, xnext_enc, tgt_key_padding_mask=xnext_pred_padding_mask, tgt_mask=cmask_xnext) # (seq_len_xnext, bs, d_model)
        # xnext_dec_seq = xnext_dec_seq.permute(1,0,2) # (bs, seq_len_xnext, d_model)
        # xnext_pred_out = self.dec_out_linear(xnext_dec_seq) # (bs, seq_len_xnext, n_tokens)

        # # loss,acc: xnext-vae
        # xnext_pred_out_ce = xnext_pred_out.permute(0,2,1) # (bs, n_tokens, seq_len_xnext)
        # loss_xnext = F.cross_entropy(xnext_pred_out_ce, xnext, ignore_index=self.pad_token_id)
        # self.log(f"{mode}_05_xnext_vae_loss", loss_xnext, prog_bar=False)
        # loss += loss_xnext
        # xnext_hat = torch.argmax(xnext_pred_out, dim=2) # (bs, seq_len_xnext)
        # acc = self.metrics[f'{mode}_xnext_vae_acc'](xnext_hat, xnext)
        # self.log(f"{mode}_05_xnext_vae_acc", acc, prog_bar=False)

        # # rowwise accuracy
        # xnext_hat_rw = torch.logical_or(xnext_hat == xnext, xnext == self.pad_token_id).all(dim=1)
        # acc = self.metrics[f'{mode}_xnext_vae_acc_rw'](xnext_hat_rw, torch.ones_like(xnext_hat_rw))
        # self.log(f"{mode}_06_xnext_vae_acc_rw", acc, prog_bar=False)
        
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
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(
                    optimizer, warmup_epochs=2000, max_epochs=100000, warmup_start_lr=0.0, eta_min=0.1*self.lr, last_epoch=-1), # torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100000, eta_min=0.1*self.lr),
                "interval": "step",
                "frequency": 1,
            },
        }


















# Define entire system as a LightningModule
class S2TransformerP2(L.LightningModule):
    def __init__(self, pt1_model, d_model, n_enc_heads, n_enc_layers, n_tokens, lr, weight_decay, pad_token_id, predictive, dropout, n_bptt, **kwargs):
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

        # freeze pt1_model
        self.pt1_model = pt1_model
        for param in self.pt1_model.parameters():
            param.requires_grad = False

        # extractor: pull out substructure
        self.extract_dec = nn.TransformerDecoder(mb.TransformerDecoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=False), num_layers=n_enc_layers)
        extract_starter = torch.rand_like(torch.zeros(d_model)) * math.sqrt(d_model)
        self.extract_starter = torch.nn.Parameter(extract_starter) # (d_model,)



        # metrics
        for mode in ['train', 'val']:
            pass
            # setattr(self, f'{mode}_x_vae_acc', torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_tokens, ignore_index=pad_token_id))
            # self.metrics[f'{mode}_x_vae_acc'] = getattr(self, f'{mode}_x_vae_acc')
            # setattr(self, f'{mode}_x_vae_acc_rw', torchmetrics.classification.Accuracy(task="binary", num_classes=2))
            # self.metrics[f'{mode}_x_vae_acc_rw'] = getattr(self, f'{mode}_x_vae_acc')

            # setattr(self, f'{mode}_x_s1_acc', torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_tokens, ignore_index=pad_token_id))
            # self.metrics[f'{mode}_x_s1_acc'] = getattr(self, f'{mode}_x_s1_acc')
            # setattr(self, f'{mode}_xt_vae_acc', torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_tokens, ignore_index=pad_token_id))
            # self.metrics[f'{mode}_xt_vae_acc'] = getattr(self, f'{mode}_xt_vae_acc')
            # setattr(self, f'{mode}_xt_s1_acc', torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_tokens, ignore_index=pad_token_id))
            # self.metrics[f'{mode}_xt_s1_acc'] = getattr(self, f'{mode}_xt_s1_acc')
            # setattr(self, f'{mode}_xnext_vae_acc', torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_tokens, ignore_index=pad_token_id))
            # self.metrics[f'{mode}_xnext_vae_acc'] = getattr(self, f'{mode}_xnext_vae_acc')

            # setattr(self, f'{mode}_xnext_vae_acc_rw', torchmetrics.classification.Accuracy(task="binary", num_classes=2))
            # self.metrics[f'{mode}_xnext_vae_acc_rw'] = getattr(self, f'{mode}_xnext_vae_acc_rw')


        self.init_weights()


    def init_weights(self) -> None:
        initrange = 0.1
        # pass
        # self.dec_out_linear.bias.data.zero_()
        # self.dec_out_linear.weight.data.uniform_(-initrange, initrange)


    def model_step(self, batch, batch_idx, mode='train'):
        height, length, x, y = batch # (bs, seq_len)
        padding_mask = (x==self.pad_token_id) # (bs, seq_len)
        x, y = x.to(torch.long), y.to(torch.long) # do conversion on GPU (mem bottleneck)
        bs, seq_len = x.shape


        # embed x
        x_emb = self.pt1_model.embedding(x.T) * math.sqrt(self.d_model) # (seq_len, bs, d_model)

        # encode long input into single token
        x_enc_seq = self.pt1_model.enc_in(x_emb, src_key_padding_mask=padding_mask) # (seq_len, bs, d_model)
        x_dec_in_starter = self.pt1_model.dec_in_starter.unsqueeze(0).unsqueeze(0).repeat(1, x_enc_seq.shape[1], 1) # (1, bs, d_model)
        x_enc = self.pt1_model.dec_in(x_dec_in_starter, x_enc_seq, memory_key_padding_mask=padding_mask) # (1, bs, d_model)

        # pull out substructure
        extract_starter = self.extract_starter.unsqueeze(0).unsqueeze(0).repeat(1, bs, 1) # (1, bs, d_model)
        x_subst = self.extract_dec(extract_starter, x_enc) # (1, bs, d_model)

        # s1-driven loss
        # push through pre-trained S1: use 1-(max_pred_prob) as loss
        # i.e. loss is zero if S1 model is 100% confident it can solve the substructure problem
        s1_subst_enc = self.pt1_model.s1_enc(x_subst) # (1, bs, d_model)
        s1_subst_pred = self.pt1_model.s1_linear(s1_subst_enc) # (1, bs, n_tokens)
        s1_subst_probs = F.softmax(s1_subst_pred, dim=2) # (1, bs, n_tokens)
        s1_subst_probs_zprob = s1_subst_probs.clone()
        s1_subst_probs_zprob[:,:,dataset.bad_token_id] = 0 # zero prob for bad_token, i.e. it silently absorbs probability mass. It reduces the max ceiling, so loss=1-maxprob will be higher it sucked out lots of probability
        s1_subst_max_probs = s1_subst_probs_zprob.max(dim=2).values.squeeze(0) # (bs,)
        loss_s1_subst = (-1*torch.log(s1_subst_max_probs)).mean()
        self.log(f"{mode}_11_loss_s1_subst", loss_s1_subst, prog_bar=True)
        loss = loss_s1_subst

        # contains-driven loss
        # encourage x_subst to be something that was contained in x_enc
        contains_subst = self.pt1_model.contains_dec(x_enc, x_subst) # (1, bs, d_model)
        contains_pred = self.pt1_model.contains_linear(contains_subst) # (1, bs, 2)
        contains_pred = F.softmax(contains_pred, dim=2) # (1, bs, 2)

        loss_contains = (-1*torch.log(contains_pred[:,:,1])).mean()
        self.log(f"{mode}_12_loss_contains", loss_contains, prog_bar=True)
        loss += loss_contains

        # self-contrastive loss
        # encourage every item in the batch to be as different as possible
        # since the input is the main source of randomness,
        # the only sustainable way for this to happen is if the model grabs larger chunks of actual substructure

        temperature = 0.1
        # normalize to have unit length
        x_subst_norm = F.normalize(x_subst, p=2, dim=2).squeeze(0) # (bs, d_model)    
        # compute similarity matrix
        x_subst_sim = torch.matmul(x_subst_norm, x_subst_norm.T) # (bs, bs)
        # create contrastive loss labels, compute contrastive loss
        labels = torch.arange(bs, dtype=torch.long).to(x_subst_sim.device) # (bs,)
        loss_contrastive = F.cross_entropy(x_subst_sim / temperature, labels)
        self.log(f"{mode}_13_loss_contrastive", loss_contrastive, prog_bar=True)
        loss += loss_contrastive

        if mode == 'val':
            n_check = 4
            if batch_idx % 100 == 0:
                in_detok = dataset.detokenize(x[0:n_check])
                print("\n")
                for in_s in in_detok:
                    print(f"Input:  {in_s}")
                print(f"Output: {y[0:n_check]}")

                # x-vae: decode: single token into long output (no teacher forcing)
                # uses s1_subst_enc, self.pt1_model.dec_out_starter, self.pt1_model.dec_out, self.pt1_model.dec_out_linear
                cmask = torch.nn.Transformer.generate_square_subsequent_mask(sz=seq_len).to(x.device) # (seq_len, seq_len)
                cur_in = self.pt1_model.dec_out_starter.unsqueeze(0).unsqueeze(0).repeat(1,n_check,1) # (1, n_check, d_model)

                # decoding loop
                for i in range(seq_len): # first 24 chars
                    cur_mask = cmask[:i+1,:i+1] # (i+1, i+1)
                    cur_dec = self.pt1_model.dec_out(cur_in, s1_subst_enc[:,:n_check], tgt_mask=cur_mask) # (i+1, n_check, d_model)
                    cur_dec = cur_dec.permute(1,0,2) # (n_check, i+1, d_model)
                    cur_pred = self.pt1_model.dec_out_linear(cur_dec[:,-1:]) # (n_check, 1, n_tokens)
                    next_tok = torch.argmax(cur_pred, dim=2) # (n_check, 1)
                    next_emb = self.pt1_model.embedding(next_tok) * math.sqrt(self.d_model) # (n_check, 1, d_model)
                    cur_in = torch.cat([cur_in, next_emb.permute(1,0,2)], dim=0) # (i+2, n_check, d_model)

                # make output
                s1_subst_dec = self.pt1_model.dec_out_linear(cur_dec) # (n_check,seq_len,n_tokens)
                s1_subst_pred = torch.argmax(s1_subst_dec, dim=2) # (n_check,seq_len)
                s1_subst_str = dataset.detokenize(s1_subst_pred)
                for out_s in s1_subst_str:
                    print(f"Substr: {out_s}")


        return loss

        # # X-S1
        # y_filtered = y[height <= 3] # (bs,)
        # x_enc_filtered = x_enc[:,height <= 3] # (1, bs_filt, d_model)

        # # x-s1: predict result over full x
        # s1_enc = self.s1_enc(x_enc_filtered) # (1, bs_filt, d_model)
        # s1_pred = self.s1_linear(s1_enc) # (1, bs_filt, n_tokens)
        # # loss,acc: x-s1
        # s1_pred_ce = s1_pred.permute(1,2,0) # (bs_filt, n_tokens, 1)
        # loss_s1 = F.cross_entropy(s1_pred_ce, y_filtered.unsqueeze(1), ignore_index=self.pad_token_id)
        # self.log(f"{mode}_02_x_s1_loss", loss_s1, prog_bar=False)
        # loss += loss_s1
        # s1_hat = torch.argmax(s1_pred, dim=2).squeeze(0) # (bs_filt,)
        # acc = self.metrics[f'{mode}_x_s1_acc'](s1_hat, y_filtered)
        # self.log(f"{mode}_02_x_s1_acc", acc, prog_bar=False)


        # # XT-VAE
        # # pull out the specific subaction (e.g. (*3(+23)) -> (+23))
        # xt_enc = self.extract_enc(x_enc) # (1, bs, d_model)
        # # delta: make a view of the state ready to be combined with the next action (e.g. (*3(+23)) -> (*3????))
        # # xt_delta_enc = self.cleancut_dec(x_enc, xt_enc) # (1, bs, d_model) 

        # # xt-decode: teacher-force extract
        # xt_emb = self.embedding(xsi.T) * math.sqrt(self.d_model) # (seq_len_xt, bs, d_model)
        # xt_pred_in = torch.cat([
        #     self.dec_out_starter.unsqueeze(0).unsqueeze(0).repeat(1, x_enc_seq.shape[1], 1), # (1, bs, d_model)
        #     xt_emb[:-1] # (seq_len_xt-1, bs, d_model),
        # ], dim=0) # (seq_len_xt, bs, d_model)
        # xt_pred_padding_mask = torch.cat([
        #     torch.zeros(x_enc_seq.shape[1], 1, dtype=torch.bool, device=x_enc_seq.device), # (bs, 1)
        #     padding_mask_xt[:,:-1] # (bs, seq_len_xt-1)
        # ], dim=1) # (bs, seq_len)
        # cmask_xt = torch.nn.Transformer.generate_square_subsequent_mask(sz=xt_emb.shape[0]).to(x.device) # (seq_len, seq_len)
        # xt_dec_seq = self.dec_out(xt_pred_in, xt_enc, tgt_key_padding_mask=xt_pred_padding_mask, tgt_mask=cmask_xt) # (seq_len_xt, bs, d_model)
        # xt_dec_seq = xt_dec_seq.permute(1,0,2) # (bs, seq_len, d_model)
        # xt_pred_out = self.dec_out_linear(xt_dec_seq) # (bs, seq_len, n_tokens)

        # # loss,acc: xt-vae
        # xt_pred_out_ce = xt_pred_out.permute(0,2,1) # (bs, n_tokens, seq_len)
        # loss_xt = F.cross_entropy(xt_pred_out_ce, xnext, ignore_index=self.pad_token_id)
        # self.log(f"{mode}_03_xt_vae_loss", loss_xt, prog_bar=False)
        # loss += loss_xt
        # xt_hat = torch.argmax(xt_pred_out, dim=2) # (bs, seq_len)
        # acc = self.metrics[f'{mode}_xt_vae_acc'](xt_hat, xnext)
        # self.log(f"{mode}_03_xt_vae_acc", acc, prog_bar=False)

        # # XT-S1
        # # xt-s1: predict result over extract
        # s1_enc_xt = self.s1_enc(xt_enc) # (1, bs, d_model)
        # s1_pred_xt = self.s1_linear(s1_enc_xt) # (1, bs, n_tokens)
        # # loss,acc: xt-s1
        # s1_pred_xt_ce = s1_pred_xt.permute(1,2,0) # (bs, n_tokens, 1)
        # loss_s1_xt = F.cross_entropy(s1_pred_xt_ce, ysi.unsqueeze(1), ignore_index=self.pad_token_id)
        # self.log(f"{mode}_04_xt_s1_loss", loss_s1_xt, prog_bar=False)
        # loss += loss_s1_xt
        # s1_hat_xt = torch.argmax(s1_pred_xt, dim=2).squeeze(0) # (bs,)
        # acc = self.metrics[f'{mode}_xt_s1_acc'](s1_hat_xt, ysi)
        # self.log(f"{mode}_04_xt_s1_acc", acc, prog_bar=False)


        # # COMBINE: delta + pred -> next state
        # ysi_enc = self.embedding(ysi.unsqueeze(0)) * math.sqrt(self.d_model) # (1, bs, d_model)
        # xnext_enc = self.combine_dec(x_enc, torch.cat([xt_enc, ysi_enc], dim=0)) # (1, bs, d_model)
        # # teacher-force decode vs xnext
        # padding_mask_xnext = (xnext==self.pad_token_id) # (bs, seq_len_xnext)
        # xnext_emb = self.embedding(xnext.T) * math.sqrt(self.d_model) # (seq_len_xnext, bs, d_model)
        # xnext_pred_in = torch.cat([
        #     self.dec_out_starter.unsqueeze(0).unsqueeze(0).repeat(1, x_enc_seq.shape[1], 1), # (1, bs, d_model)
        #     xnext_emb[:-1] # (seq_len_xnext-1, bs, d_model),
        # ], dim=0)
        # xnext_pred_padding_mask = torch.cat([
        #     torch.zeros(x_enc_seq.shape[1], 1, dtype=torch.bool, device=x_enc_seq.device), # (bs, 1)
        #     padding_mask_xnext[:,:-1] # (bs, seq_len_xnext-1)
        # ], dim=1)
        # cmask_xnext = torch.nn.Transformer.generate_square_subsequent_mask(sz=xnext_emb.shape[0]).to(x.device) # (seq_len_xnext, seq_len_xnext)
        # xnext_dec_seq = self.dec_out(xnext_pred_in, xnext_enc, tgt_key_padding_mask=xnext_pred_padding_mask, tgt_mask=cmask_xnext) # (seq_len_xnext, bs, d_model)
        # xnext_dec_seq = xnext_dec_seq.permute(1,0,2) # (bs, seq_len_xnext, d_model)
        # xnext_pred_out = self.dec_out_linear(xnext_dec_seq) # (bs, seq_len_xnext, n_tokens)

        # # loss,acc: xnext-vae
        # xnext_pred_out_ce = xnext_pred_out.permute(0,2,1) # (bs, n_tokens, seq_len_xnext)
        # loss_xnext = F.cross_entropy(xnext_pred_out_ce, xnext, ignore_index=self.pad_token_id)
        # self.log(f"{mode}_05_xnext_vae_loss", loss_xnext, prog_bar=False)
        # loss += loss_xnext
        # xnext_hat = torch.argmax(xnext_pred_out, dim=2) # (bs, seq_len_xnext)
        # acc = self.metrics[f'{mode}_xnext_vae_acc'](xnext_hat, xnext)
        # self.log(f"{mode}_05_xnext_vae_acc", acc, prog_bar=False)

        # # rowwise accuracy
        # xnext_hat_rw = torch.logical_or(xnext_hat == xnext, xnext == self.pad_token_id).all(dim=1)
        # acc = self.metrics[f'{mode}_xnext_vae_acc_rw'](xnext_hat_rw, torch.ones_like(xnext_hat_rw))
        # self.log(f"{mode}_06_xnext_vae_acc_rw", acc, prog_bar=False)


        




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
