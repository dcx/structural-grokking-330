# this is a version of model.py which attempts to parallelize the transformer VAE encoding and decoding
# it doesn't work - trains to 100%, but when you try to decode, it totally fails
# suspect there is leakage between x_enc tokens, or something about the way we're averaging,
# or rotary position embeddings are messing with x_enc

import torch, torch.nn as nn, torch.utils.data as data, torch.nn.functional as F
import lightning as L
import math
import torchmetrics
import random
import modelbasics as mb


# Define entire system as a LightningModule
class BPTTTransformer(L.LightningModule):
    """
    A LightningModule (nn.Module subclass) defines a full *system*
    (ie: an LLM, diffusion model, autoencoder, or simple image classifier).
    """

    def __init__(self, d_model, n_enc_heads, n_enc_layers, n_unique_tokens, n_output_tokens, lr, weight_decay, pad_token_id, max_steps):
        super().__init__()
        self.save_hyperparameters()

        self.d_model = d_model
        self.lr = lr
        self.wd = weight_decay
        self.pad_token_id = pad_token_id
        self.max_steps = max_steps

        # transformer-based VAE
        self.enc = nn.TransformerEncoder(mb.TransformerEncoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu'), num_layers=n_enc_layers)
        self.dec = nn.TransformerDecoder(mb.TransformerDecoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu'), num_layers=n_enc_layers)
        self.linear = nn.Linear(d_model, n_output_tokens)
        self.embedding = nn.Embedding(n_unique_tokens, d_model, padding_idx=pad_token_id)

        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_unique_tokens, ignore_index=pad_token_id)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_unique_tokens, ignore_index=pad_token_id)
        self.train_rowwise_acc = torchmetrics.classification.Accuracy(task="binary", num_classes=2)
        self.val_rowwise_acc = torchmetrics.classification.Accuracy(task="binary", num_classes=2)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward_enc(self, x: torch.Tensor, padding_mask: torch.Tensor=None) -> torch.Tensor:
        # embed x
        x_emb = self.embedding(x.T) * math.sqrt(self.d_model) # (seq_len, bs, d_model)

        # push through causal masked encoder
        mask = torch.nn.Transformer.generate_square_subsequent_mask(sz=x_emb.shape[0]).to(x.device) # (seq_len, seq_len)
        x_enc = self.enc(x_emb, mask=mask, src_key_padding_mask=padding_mask, is_causal=True) # (seq_len, bs, d_model)

        return x_emb, x_enc

    def forward_dec(self, x_tf_in, x_enc: torch.Tensor, padding_mask: torch.Tensor=None) -> torch.Tensor:
        """
        x_enc: (seq_len, bs, d_model)
        """
        # push through decoder
        # tricky: we're trying to encourage an x_enc which encodes the entire input
        # within a single token, so we need unusual masking
        
        # tgt_mask: standard causal mask (word 3 can see words 1 and 2)
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(sz=x_enc.shape[0]).to(x_enc.device) # (seq_len, seq_len)

        # tgt_key_padding_mask: mask out padding tokens: add unmasked first col, and offset
        tgt_key_padding_mask = torch.cat( # (bs, seq_len)
            [torch.zeros((x_enc.shape[1], 1)).to(x_enc.device), # (bs, 1)
             padding_mask[:, :-1]], dim=1).bool() # (bs, seq_len-1)
        
        # produce diagonally averaged x_enc, so that: 
        # - the first token can see x_enc's for all timesteps
        # - the second token can see x_enc's for all but the first timestep
        # - etc. (seq_len, bs, d_model)
        # so that the decoder can learn to predict the entire input from a single token

        # zero x_enc for padding tokens
        x_enc = x_enc * (~(padding_mask.T)).unsqueeze(2) # (seq_len, bs, d_model)
        # project out: (seq_len, bs, d_model) -> (seq_len, seq_len, bs, d_model)
        x_enc_projected = x_enc.unsqueeze(0).repeat(x_enc.shape[0], 1, 1, 1) # (seq_len, seq_len, bs, d_model)
        # flip for convenience:
        x_enc_projected = x_enc_projected.permute(2, 3, 1, 0) # (bs, d_model, seq_len, seq_len)
        # apply triangular mask, and flip so first token can see all
        x_enc_proj_masked_a = torch.tril(x_enc_projected, diagonal=0) # (bs, d_model, seq_len, seq_len)
        # x_enc_proj_masked_b = torch.flip(x_enc_proj_masked_a, dims=(3,)) # (bs, d_model, seq_len, seq_len)
        # sum over diagonals
        x_enc_proj_masked_c = torch.sum(x_enc_proj_masked_a, dim=2) # (bs, d_model, seq_len)

        # count number of elements in each diagonal
        # (we want to divide by this number, to get the average)
        n_seq_elements = torch.sum((~padding_mask),dim=1).unsqueeze(1).unsqueeze(2) # (bs, 1, 1)

        # divide by number of elements in diagonal
        x_enc_proj_masked_d = x_enc_proj_masked_c / n_seq_elements # (bs, d_model, seq_len)
            # / torch.arange(x_enc_proj_masked_c.shape[2], 0, -1).to(x_enc.device) # (bs, d_model, seq_len)
        # restore x_enc dims
        x_enc_proj_masked = torch.permute(x_enc_proj_masked_d, (2, 0, 1)) # (seq_len, bs, d_model)

        # # memory_mask (T,S): 
        # # - triangular: every word in the target decoder sequence can attend to the same word in source memory
        # # - but decoder word 3 is only encoded in x_enc >=3, not x_enc 1,2
        # # memory_mask = torch.triu(torch.ones(x_enc.shape[0], x_enc.shape[0]), diagonal=1).to(x_enc.device).bool() # (seq_len, seq_len)
        # memory_mask = ~torch.triu(torch.ones(x_enc.shape[0], x_enc.shape[0]), diagonal=0).to(x_enc.device).bool() # (seq_len, seq_len)

        # memory mask (T,S): diagonal: at word 3, look at only x_enc #3
        # the combining we do above packs the right vars into each x_enc item
        # (the idea being, we want to predict the entire input from a single token)
        memory_mask = torch.diag(torch.ones(x_enc.shape[0])).to(x_enc.device).bool() # (seq_len, seq_len)
        memory_mask = ~memory_mask # (seq_len, seq_len)
        #memory_key_padding_mask = padding_mask # (bs, seq_len)

        x_dec = self.dec(x_tf_in, x_enc_proj_masked, 
                         tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_mask=memory_mask, #memory_key_padding_mask=padding_mask,
                         tgt_is_causal=True) # (seq_len, bs, d_model)

        # project out to n_output_tokens
        x_dec = self.linear(x_dec) # (seq_len, bs, n_output_tokens)

        return x_dec

        

        # # initialize wm with zeros
        # wm_current = torch.zeros((x.shape[1], x.shape[0], self.d_model)).to(x.device) # (seq_len, bs, d_model)

        # for i in range(n_steps):
        #     # run s1
        #     s1_proposed = self.s1(x, wm_current, padding_mask) # (seq_len*2, bs, d_model)
        #     # run s2 and update wm
        #     wm_current = self.s2(s1_proposed, wm_current, padding_mask) # (seq_len, bs, d_model)

        # # run s2o on final step
        # output_pred = self.s1o(s1_proposed, padding_mask) # (seq_len, bs, n_output_tokens)
        # return output_pred # (seq_len, bs, ntoken)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        b, padding_mask = batch # (bs, max_steps, seq_len)

        # pick a random number of steps, from 2 to max_steps
        n_steps = 1 # random.randint(2, self.max_steps)

        # x is first step, y is n_steps later
        x = b[:, 0, :] # (bs, seq_len)
        # y = b[:, n_steps, :] # (bs, seq_len)
        padding_mask = padding_mask[:, 0, :] # (bs, seq_len)

        # encode
        x_emb, x_enc = self.forward_enc(x, padding_mask=padding_mask) # (seq_len, bs, d_model)

        # prepare teacher-forcing input x: shifted with start of sequence token
        x_tf_in = torch.cat(
            [torch.ones((1, x_emb.shape[1], x_emb.shape[2])).to(x.device), # (1, bs, d_model)
             x_emb[:-1, :, :]], dim=0) # (seq_len, bs, d_model)

        # decode
        y_hat = self.forward_dec(x_tf_in, x_enc, padding_mask=padding_mask) # (seq_len, bs, n_output_tokens)

        # loss
        ce_y_hat = torch.permute(y_hat, (1, 2, 0)) # (bs, n_output_tokens, seq_len)
        # scale cross-entropy sequence position
        # so that the averaging doesn't unfairly penalize longer sequences
        ce_y_hat = ce_y_hat * torch.arange(ce_y_hat.shape[2], 0, -1).to(x.device).unsqueeze(0).unsqueeze(1) # (bs, n_output_tokens, seq_len)

    
        loss = F.cross_entropy(ce_y_hat, x, ignore_index=self.pad_token_id)
        self.log("train_loss", loss)

        # accuracy
        y_pred = torch.argmax(y_hat, dim=2).T # (bs, seq_len)
        self.train_acc(y_pred, x)
        self.log('train_acc_step', self.train_acc)

        # rowwise accuracy: do we get the whole row right?
        y_match = (y_pred == x) | (x == self.pad_token_id)
        y_match_row = y_match.all(dim=1).long() # (bs,)
        self.train_rowwise_acc(y_match_row, torch.ones_like(y_match_row))
        self.log('train_rowwise_acc_step', self.train_rowwise_acc)


        return loss
    
    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        b, padding_mask = batch # (bs, max_steps, seq_len)

        # pick a random number of steps, from 2 to max_steps
        n_steps = 1 # random.randint(2, self.max_steps)

        # x is first step, y is n_steps later
        x = b[:, 0, :] # (bs, seq_len)
        # y = b[:, n_steps, :] # (bs, seq_len)
        padding_mask = padding_mask[:, 0, :] # (bs, seq_len)

        # encode
        x_emb, x_enc = self.forward_enc(x, padding_mask=padding_mask) # (seq_len, bs, d_model)

        # prepare teacher-forcing input x: shifted with start of sequence token
        x_tf_in = torch.cat(
            [torch.ones((1, x_emb.shape[1], x_emb.shape[2])).to(x.device), # (1, bs, d_model)
             x_emb[:-1, :, :]], dim=0) # (seq_len, bs, d_model)

        # decode
        y_hat = self.forward_dec(x_tf_in, x_enc, padding_mask=padding_mask) # (seq_len, bs, n_output_tokens)

        # loss
        ce_y_hat = torch.permute(y_hat, (1, 2, 0)) # (bs, n_output_tokens, seq_len)
        loss = F.cross_entropy(ce_y_hat, x, ignore_index=self.pad_token_id)
        self.log("val_loss", loss)

        # accuracy
        y_pred = torch.argmax(y_hat, dim=2).T # (bs, seq_len)
        self.val_acc(y_pred, x)
        self.log('val_acc_step', self.train_acc)

        # rowwise accuracy: do we get the whole row right?
        y_match = (y_pred == x) | (x == self.pad_token_id)
        y_match_row = y_match.all(dim=1).long() # (bs,)
        self.val_rowwise_acc(y_match_row, torch.ones_like(y_match_row))
        self.log('val_rowwise_acc_step', self.val_rowwise_acc)

        return loss

    def decode(self, x_enc, x):
        """
        Non teacher-forced decoding of a single encoded sequence.

        x_enc: (d_model,)
        x: (seq_len,)
        returns: (seq_len, n_output_tokens)

        self.decode(x_enc[9,0], x[0])
        """
        print(f"x:{x.tolist()}")

        # prepare teacher-forcing input x: shifted with start of sequence token
        x_wip = torch.ones((1, 1, self.d_model)).to(x_enc.device) # (1, 1, d_model)

        # decode
        x_enc_shaped = x_enc.unsqueeze(0).unsqueeze(0) # (1, 1, d_model)
        
        # without teacher forcing
        for i in range(self.max_steps):
            # project pxadding mask and encode
            padding_mask = torch.zeros((x_wip.shape[1], x_wip.shape[0]), dtype=torch.bool).to(x_enc.device) # (bs, seq_len)
            x_enc_proj = x_enc_shaped.repeat(x_wip.shape[0], x_wip.shape[1], 1) # (seq_len, bs, d_model)

            x_logits = self.forward_dec(x_wip, x_enc_proj, padding_mask=padding_mask) # (seq_len, 1, n_output_tokens)
            x_preds = torch.argmax(x_logits, dim=2) # (seq_len, 1)

            # prepare next input
            print(f"{i}:{x_preds.tolist()}")            
            x_pred_embed = self.embedding(x_preds) * math.sqrt(self.d_model) # (seq_len, 1, d_model)
            x_wip = torch.cat([x_wip, x_pred_embed[-1:]], dim=0)

        # with teacher forcing (same as above)
        x_emb = self.embedding(x.unsqueeze(1).T) * math.sqrt(self.d_model) # (seq_len, 1, d_model)
        x_tf_in = torch.cat(
            [torch.ones((1, x_emb.shape[1], x_emb.shape[2])).to(x.device), # (1, bs, d_model)
             x_emb[:-1, :, :]], dim=0) # (seq_len, 1, d_model)

        padding_mask = torch.zeros((x_emb.shape[1], x_emb.shape[0]), dtype=torch.bool).to(x_enc.device) # (bs, seq_len)
        # set True where x==pad_token_id
        padding_mask[x==self.pad_token_id] = True
        x_enc_proj = x_enc_shaped.repeat(x_emb.shape[0], x_emb.shape[1], 1) # (seq_len, bs, d_model)
        y_hat = self.forward_dec(x_tf_in, x_enc_proj, padding_mask=padding_mask) # (seq_len, bs, n_output_tokens)
        y_pred = torch.argmax(y_hat, dim=2).T # (bs, seq_len)
        print(y_pred)





        

    def on_train_epoch_end(self):
        self.log('train_acc_epoch', self.train_acc.compute())
        self.train_acc.reset()
        self.log('train_rowwise_acc_epoch', self.train_rowwise_acc.compute())
        self.train_rowwise_acc.reset()

    def on_val_epoch_end(self):
        self.log('val_acc_epoch', self.val_acc.compute())
        self.val_acc.reset()
        self.log('val_rowwise_acc_epoch', self.val_rowwise_acc.compute())
        self.val_rowwise_acc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer





if __name__ == "__main__":
    
    # shake out end-to-end
    d_model = 64
    n_enc_heads = 2
    n_enc_layers = 2
    n_unique_tokens = 5 # (0,1,ca_rule_0,ca_rule_1,pad)
    n_output_tokens = 2 # (0,1)
    pad_token_id = 4

    s1 = System1(d_model, n_enc_heads, n_enc_layers, n_unique_tokens, pad_token_id)
    s2 = System2(d_model, n_enc_heads, n_enc_layers, pad_token_id)
    s2o = S1OutputTranslator(d_model, n_enc_heads, n_enc_layers, pad_token_id, n_output_tokens)

    # init dummy starting wm
    dummy_seq_len = 6
    dummy_bs = 2
    wm_current = torch.randn((dummy_seq_len, dummy_bs, d_model)) # (seq_len, bs, d_model)

    # init dummy input
    x_raw = torch.randint(0, n_unique_tokens, (dummy_bs, dummy_seq_len)) # (bs, seq_len)
    padding_mask = torch.zeros((dummy_bs, dummy_seq_len), dtype=torch.bool) # (bs, seq_len)

    # run s1
    s1_proposed = s1(x_raw, wm_current, padding_mask) # (seq_len*2, bs, d_model)

    # run s2
    wm_next = s2(s1_proposed, wm_current, padding_mask) # (seq_len, bs, d_model)

    # run s2o
    output_pred = s2o(s1_proposed, padding_mask) # (seq_len, bs, n_output_tokens)

