import torch, torch.nn as nn, torch.utils.data as data, torch.nn.functional as F
import lightning as L
import math
import torchmetrics
import random
import modelbasics as mb


# Define entire system as a LightningModule
class S1Transformer(L.LightningModule):
    """
    A LightningModule (nn.Module subclass) defines a full *system*
    (ie: an LLM, diffusion model, autoencoder, or simple image classifier).

    This is the module for System 1:
    a transformer-based VAE that creates a latent representation of the input,
    and uses that to predict the next step in the sequence.

    WARNING: Still rough around the edges:
    - Parallelization is still imperfect (note the double for loop)
    - Requires rotary position embeddings hack in functional.py
    - I suspect there's a small bug around padding and x_enc's
      (i.e. there may be some pad vals leaking somewhere, e.g. memory_mask)

    Things worth trying
    - Use a single model for both VAE and predictive segments
    - Try not using teacher forcing, but instead doing reconstruction
      using all ones on the decoder input 
      (early sanity check shows promise: https://wandb.ai/dcx/lightning_logs/runs/aq2dikll)
    - Figure out how to make the representation >1 tokens long
      Otherwise n_model_dims is too high, may create rotary embedding / precision issues
    """

    def __init__(self, d_model, n_enc_heads, n_enc_layers, n_tokens, lr, weight_decay, pad_token_id, predictive, dropout):
        super().__init__()
        self.save_hyperparameters()

        self.d_model = d_model
        self.lr = lr
        self.wd = weight_decay
        self.pad_token_id = pad_token_id
        self.predictive = predictive
        self.metrics = {}
        self.dropout = dropout

        # transformer-based VAE
        self.enc = nn.TransformerEncoder(mb.TransformerEncoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=True), num_layers=n_enc_layers)
        self.dec = nn.TransformerDecoder(mb.TransformerDecoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=True), num_layers=n_enc_layers)
        self.linear = nn.Linear(d_model, n_tokens)
        self.embedding = nn.Embedding(n_tokens, d_model, padding_idx=pad_token_id)
        self.dummy_first_token = (torch.rand_like(torch.zeros((1, 1, d_model))) * math.sqrt(d_model)).to('cuda') # (1, 1, d_model)
        # TODO: make this a parameter

        for mode in ['train', 'val']:
            # hack: metrics must be on self or Lightning doesn't handle their devices correctly
            setattr(self, f'{mode}_vae_acc', torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_tokens, ignore_index=pad_token_id))
            setattr(self, f'{mode}_vae_rowwise_acc', torchmetrics.classification.Accuracy(task="binary", num_classes=2))
            self.metrics[f'{mode}_vae_acc'] = getattr(self, f'{mode}_vae_acc')
            self.metrics[f'{mode}_vae_rowwise_acc'] = getattr(self, f'{mode}_vae_rowwise_acc')

        if self.predictive:
            self.pred_enc = nn.TransformerEncoder(mb.TransformerEncoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout, batch_first=True), num_layers=n_enc_layers)
            # self.pred_dec = nn.TransformerDecoder(mb.TransformerDecoderLayer(d_model=d_model, nhead=n_enc_heads, activation='gelu', dropout=self.dropout), num_layers=n_enc_layers)
            self.pred_linear = nn.Linear(d_model, n_tokens)
            for mode in ['train', 'val']:
                # hack: metrics must be on self or Lightning doesn't handle their devices correctly
                setattr(self, f'{mode}_pred_acc', torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_tokens, ignore_index=pad_token_id))
                #setattr(self, f'{mode}_pred_rowwise_acc', torchmetrics.classification.Accuracy(task="binary", num_classes=2))
                self.metrics[f'{mode}_pred_acc'] = getattr(self, f'{mode}_pred_acc')
                #self.metrics[f'{mode}_pred_rowwise_acc'] = getattr(self, f'{mode}_pred_rowwise_acc')

            self.dummy_input = (torch.rand_like(torch.zeros((24, 1, d_model))) * math.sqrt(d_model)).to('cuda') # (24, 1, d_model)
            # TODO: make this a parameter

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)
        if self.predictive:
            self.pred_linear.bias.data.zero_()
            self.pred_linear.weight.data.uniform_(-initrange, initrange)

    def forward_enc(self, x: torch.Tensor, padding_mask: torch.Tensor=None) -> torch.Tensor:
        # embed x
        x_emb = self.embedding(x.T) * math.sqrt(self.d_model) # (seq_len, bs, d_model)

        # push through causal masked encoder
        mask = torch.nn.Transformer.generate_square_subsequent_mask(sz=x_emb.shape[0]).to(x.device) # (seq_len, seq_len)
        x_enc = self.enc(x_emb.transpose(1,0), mask=mask, src_key_padding_mask=padding_mask, is_causal=True) # (bs, seq_len, d_model)
        x_enc = x_enc.transpose(1,0) # (bs, seq_len, d_model) HACK: plug in batch_first quickly for FlashAttention

        # zero x_enc for padding tokens
        x_enc = x_enc * (~(padding_mask.T)).unsqueeze(2)

        return x_emb, x_enc # x_enc_final

    def forward_dec(self, x_tf_in, x_enc: torch.Tensor, padding_mask: torch.Tensor=None) -> torch.Tensor:
        """
        simplified version to prove concept: 
        calculate the combined matrix in a for loop

        x_enc: (seq_len, bs, d_model)
        padding_mask: (bs, seq_len)
        """
        # # push through decoder
        # seq_len, bs, d_model = x_enc.shape

        # # # get number of non-pad timesteps of x_enc, by batch and by sequence
        # n_enc_steps_batch = torch.sum(~padding_mask, dim=1) # (bs,)

        # # build x_enc_parallel: random sum of all encodings which should be able to reconstruct x

        # # make mesh of random x_enc indices to average
        # x_enc_mesh = torch.randn(padding_mask.shape, device=x_enc.device) > 0.2 # (bs, seq_len)
        # x_enc_mesh &= ~padding_mask # set False for padding tokens

        # # always select the last non-pad step of each batch, so there's no divide-by-zero
        # x_enc_mesh[torch.arange(bs), n_enc_steps_batch-1] = True

        # # x_enc_denom: cumulative #non-pad steps to average, from right to left
        # x_enc_denom = torch.cumsum(x_enc_mesh.fliplr(), dim=1).fliplr() # (bs, seq_len)
        # # set 1 at padding tokens, so we don't divide by zero
        # x_enc_denom[padding_mask] = 1

        # # x_enc_num: cumulative sum of selected x_enc steps, from right to left
        # # (at step j, any x_enc from steps >= j should know how to reconstruct that step)
        # x_enc_mesh_steps = x_enc.transpose(1,0) * x_enc_mesh.unsqueeze(2) # (bs, seq_len, d_model)
        # x_enc_num = torch.cumsum(x_enc_mesh_steps.flip(dims=[1]), dim=1).flip(dims=[1]) # (bs, seq_len, d_model)

        # x_enc_parallel = x_enc_num / x_enc_denom.unsqueeze(2) # (bs, seq_len, d_model)
        # x_enc_parallel = x_enc_parallel.transpose(1,0) # (seq_len, bs, d_model)

        # TODO: The above is mildly suspicious - early x_enc tokens never get tested in isolation
        # and the last token is always present


        # # setup x_enc_parallel
        # n_enc_steps_batch = torch.sum(~padding_mask, dim=1) # (bs,)
        # x_enc_parallel = torch.zeros_like(x_enc) # (seq_len, bs, d_model)
        # # for i in range(x_enc.shape[1]): # for each batch
        # #     n_valid_steps = n_enc_steps_batch[i]

        # max_step_batch = n_enc_steps_batch.max().item()
        # for j in range(max_step_batch): # for each timestep
        #     # choose random subset of x_enc from [j:n_valid_steps], average
        #     n_possible_steps = n_enc_steps_batch - j # (bs,)
        #     n_steps_to_average = torch.randn(n_possible_steps.shape, device=x_enc.device) # (bs,) [0,1]
        #     n_steps_to_average = (n_possible_steps.float() * n_steps_to_average).long() # (bs,) [0,n_possible_steps]

        #     steps_to_average = random.sample(range(j, n_valid_steps), n_steps_to_average)
        #     x_enc_parallel[j,i,:] = torch.mean(x_enc[steps_to_average,i,:], dim=0) # (d_model,)


        # # setup x_enc_parallel
        # # TODO: Improve parallelization (but it's not the bottleneck)
        # n_enc_steps_batch = torch.sum(~padding_mask, dim=1) # (bs,)
        # x_enc_parallel = torch.zeros_like(x_enc) # (seq_len, bs, d_model)
        # for i in range(x_enc.shape[1]): # for each batch
        #     n_valid_steps = n_enc_steps_batch[i]
        #     for j in range(n_valid_steps): # for each timestep
        #         # choose random subset of x_enc from [j:n_valid_steps], average
        #         n_possible_steps = n_valid_steps - j
        #         n_steps_to_average = random.randint(1, n_possible_steps)
        #         steps_to_average = random.sample(range(j, n_valid_steps), n_steps_to_average)
        #         x_enc_parallel[j,i,:] = torch.mean(x_enc[steps_to_average,i,:], dim=0) # (d_model,)






        # setup x_enc_parallel
        # TODO: Improve parallelization (but it's not the bottleneck)
        seq_len, bs, d_model = x_enc.shape
        n_enc_steps_batch = torch.sum(~padding_mask, dim=1) # (bs,)
        min_steps = torch.min(n_enc_steps_batch) # number
        x_enc_parallel = torch.zeros_like(x_enc) # (seq_len, bs, d_model)

        rand_floor = torch.arange(min_steps, device=x_enc.device)
        rand_ceiling = torch.ones_like(rand_floor) * min_steps

        # WIP
        # rand_floor = torch.arange(min_steps, device=x_enc.device).unsqueeze(0).repeat(bs,1) # (bs, min_steps)
        # rand_ceiling = torch.ones_like(rand_floor) * min_steps

        rand_indices = torch.randint(2**63 - 1, size=(4,min_steps), device=x_enc.device) % (rand_ceiling - rand_floor) + rand_floor # (4, min_steps)
        x_enc_parallel[:min_steps,:,:] = torch.mean(x_enc[rand_indices,:,:], dim=0) # (seq_len, bs, d_model)
        x_enc_parallel[min_steps:,:,:] = x_enc[min_steps:,:,:] # (seq_len, bs, d_model)


        # for i in range(x_enc.shape[1]): # for each batch
        #     n_valid_steps = n_enc_steps_batch[i]



        # # make a selection grid for x_enc_parallel
        # seq_len, bs, d_model = x_enc.shape
        # select_grid = torch.ones(seq_len, seq_len).to(x_enc.device).bool().triu() # (seq_len, seq_len)
        # select_grid = select_grid.unsqueeze(0).repeat(bs, 1, 1) # (bs, seq_len, seq_len): word 0 can use enc 0+

        # select_pads =  (~padding_mask).unsqueeze(2).repeat(1, 1, seq_len) # (bs, seq_len, seq_len)
        # select_pads &= (~padding_mask).unsqueeze(1).repeat(1, seq_len, 1) # (bs, seq_len, seq_len)

        # select_grid &= select_pads # (bs, seq_len, seq_len)
        # select_grid = select_grid.to(torch.int8) # (bs, seq_len, seq_len)
        # select_counts = torch.sum(select_grid, dim=2) # (bs, seq_len)
        # select_counts[padding_mask] = 1 # (bs, seq_len): avoid divide-by-zero

        # select_probs = select_grid.to(torch.bfloat16) / select_counts.unsqueeze(2).to(torch.bfloat16) # (bs, seq_len, seq_len)
        # select_probs = torch.clamp(select_grid*2.0, max=1.0) # (bs, seq_len, seq_len)
        
        # select_items = torch.bernoulli(select_probs) # (bs, seq_len, seq_len)

        # # if nothing selected, add current step's encoding
        # select_counts = torch.sum(select_items, dim=2) # (bs, seq_len)
        # add_eye_select = (select_counts == 0) & ~padding_mask # (bs, seq_len)
        # select_items[:,torch.arange(seq_len),torch.arange(seq_len)] += add_eye_select

        # select_counts = torch.sum(select_items, dim=2) # (bs, seq_len) update
        # select_counts[padding_mask] = 1 # (bs, seq_len): avoid divide-by-zero

        # # apply selection grid to x_enc_parallel
        # x_enc_parallel = x_enc.permute(1,0,2).unsqueeze(2).repeat(1,1,seq_len,1) # (bs, seq_len, seq_len, d_model)
        # x_enc_parallel *= select_items.unsqueeze(3) # (bs, seq_len, seq_len, d_model)
        # x_enc_parallel = torch.sum(x_enc_parallel, dim=2) # (bs, seq_len, d_model)
        # x_enc_parallel /= select_counts.unsqueeze(2) # (bs, seq_len, d_model)
        # x_enc_parallel = x_enc_parallel.permute(1,0,2) # (seq_len, bs, d_model)


        # setup masks
        # tgt_mask: standard causal mask (word 3 can see words 1 and 2)
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(sz=x_tf_in.shape[0]).to(x_tf_in.device) # (seq_len, seq_len)
        # tgt_key_padding_mask: mask out padding tokens: add unmasked first col, and offset
        tgt_key_padding_mask = torch.cat( # (bs, seq_len)
            [torch.zeros((x_enc.shape[1], 1)).to(x_enc.device), # (bs, 1)
             padding_mask[:, :-1]], dim=1).bool() # (bs, seq_len-1)

        # memory mask (T,S): 
        # every word in the target decoder sequence gets exactly one word in the source memory
        # (which we've prepared to be an average of a random subset of x_enc's)
        memory_mask = torch.diag(torch.ones(x_enc.shape[0])).to(x_enc.device).bool() # (seq_len, seq_len)
        memory_mask = ~memory_mask # (seq_len, seq_len)

        # run decoder
        x_tf_in_bf = x_tf_in.transpose(1,0) # (bs, seq_len, d_model)
        x_enc_parallel_bf = x_enc_parallel.transpose(1,0) # (bs, seq_len, d_model)
        x_dec = self.dec(x_tf_in_bf, x_enc_parallel_bf, 
                         tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_mask=memory_mask,
                         tgt_is_causal=True) # (seq_len, bs, d_model)
        x_dec = x_dec.transpose(1,0) # (bs, seq_len, d_model) HACK: plug in batch_first quickly for FlashAttention

        # project out to n_tokens
        x_dec = self.linear(x_dec) # (seq_len, bs, n_tokens)

        return x_dec


    def model_step(self, batch, batch_idx, mode='train'):
        # training_step defines the train loop. It is independent of forward
        x, y = batch # (bs, seq_len)
        padding_mask = (x==self.pad_token_id) # (bs, seq_len)
        x, y = x.to(torch.long), y.to(torch.long) # do conversion on GPU (mem bottleneck)


        # VAE SEGMENT

        # encode x into x_enc
        x_emb, x_enc = self.forward_enc(x, padding_mask=padding_mask) # (seq_len, bs, d_model)

        # prepare teacher-forcing input x: shifted with start of sequence token
        first_tokens = self.dummy_first_token.repeat(1, x_emb.shape[1], 1) # (1, bs, d_model)
        x_tf_in = torch.cat([first_tokens, x_emb[:-1, :, :]], dim=0) # (seq_len, bs, d_model)

        # decode x_enc into x_hat
        x_hat = self.forward_dec(x_tf_in, x_enc, padding_mask=padding_mask) # (seq_len, bs, n_tokens)

        # loss
        ce_x_hat = torch.permute(x_hat, (1, 2, 0)) # (bs, n_tokens, seq_len)

        # (TODO: Decide if we still want to do the below)
        # scale cross-entropy sequence position
        # so that the averaging doesn't unfairly penalize longer sequences
        # bug: scaling factor is not always [17 16 15 14...] - some sequences are shorter
        # ce_x_hat = ce_x_hat * torch.arange(ce_x_hat.shape[2], 0, -1).to(x.device).unsqueeze(0).unsqueeze(1) # (bs, n_tokens, seq_len)

        loss = F.cross_entropy(ce_x_hat, x, ignore_index=self.pad_token_id)
        self.log(f"{mode}_vae_loss", loss)

        # accuracy
        x_pred = torch.argmax(x_hat, dim=2).T # (bs, seq_len)
        self.metrics[f'{mode}_vae_acc'](x_pred, x)
        self.log(f'{mode}_vae_acc_step', self.metrics[f'{mode}_vae_acc'])
        # rowwise accuracy: do we get the whole row right?
        x_match = (x_pred == x) | (x == self.pad_token_id)
        x_match_row = x_match.all(dim=1).long() # (bs,)
        self.metrics[f'{mode}_vae_rowwise_acc'](x_match_row, torch.ones_like(x_match_row))
        self.log(f'{mode}_vae_rowwise_acc_step', self.metrics[f'{mode}_vae_rowwise_acc'])

        # if mode == 'val':
        #     # decode sanity check
        #     self.decode(x_enc[10,0], x[0])

        if self.predictive:
            # PREDICTIVE SEGMENT

            # longify: since each x_enc token represents an entire sequence,
            # predict with a single token per batch: (1, bs*seq_len, d_model).
            # this isn't strictly necessary, but's cleaner to reason over
            seq_len, bs, d_model = x_enc.shape
            x_enc_long = x_enc.reshape(-1, 1, d_model) # (bs*seq_len, 1, d_model)
            pad_long = padding_mask.reshape(-1, 1) # (bs*seq_len, 1)

            # make prediction
            y_pred_enc_long = self.pred_enc(x_enc_long) # (bs*seq_len, 1, d_model)
            y_pred = self.pred_linear(y_pred_enc_long) # (bs*seq_len, 1, n_tokens)
            y_pred = y_pred.reshape(seq_len, bs, -1) # (seq_len, bs, n_tokens)
            y_pred = y_pred.permute(1, 0, 2) # (bs, seq_len, n_tokens)

            # loss
            ce_y_pred = torch.permute(y_pred, (0, 2, 1)) # (bs, n_tokens, seq_len)
            loss_pred = F.cross_entropy(ce_y_pred, y, ignore_index=self.pad_token_id)
            self.log(f"{mode}_pred_loss", loss_pred)
            # accuracy
            y_hat = torch.argmax(y_pred, dim=2) # (bs, seq_len)
            self.metrics[f'{mode}_pred_acc'](y_hat, y)
            self.log(f"{mode}_pred_acc_step", self.metrics[f'{mode}_pred_acc'])
            # rowwise accuracy is meaningless for next-word prediction
            # (each word's pred is coming from a different encoding)
            loss += loss_pred

        return loss

    def training_step(self, batch, batch_idx):
        return self.model_step(batch, batch_idx, mode='train')
    
    def validation_step(self, batch, batch_idx):
        return self.model_step(batch, batch_idx, mode='val')

    def decode(self, x_enc_i, x_i):
        """
        Non teacher-forced decoding of a single encoded sequence.

        x_enc_i: (d_model,)
        x_i: (seq_len,)
        returns: (seq_len, n_tokens)

        Usage: self.decode(x_enc[9,0], x[0])
        """
        print(f"x:{x_i.tolist()}")

        # prepare teacher-forcing input x: shifted with start of sequence token
        x_wip = self.dummy_first_token # (1, 1, d_model)

        # decode
        x_enc = x_enc_i.unsqueeze(0).unsqueeze(1) # (1, 1, d_model)
        
        for i in range(10):
            # without teacher forcing
            padding_mask = torch.zeros((x_wip.shape[1], x_wip.shape[0]), dtype=torch.bool).to(x_enc_i.device) # (bs, seq_len)
            x_enc_proj = x_enc.repeat(x_wip.shape[0], x_wip.shape[1], 1) # (seq_len, bs, d_model)

            x_logits = self.forward_dec(x_wip, x_enc_proj, padding_mask=padding_mask) # (seq_len, 1, n_tokens)
            x_preds = torch.argmax(x_logits, dim=2) # (seq_len, 1)

            # prepare next input
            print(f"{i}:{x_preds.tolist()}")            
            x_pred_embed = self.embedding(x_preds) * math.sqrt(self.d_model) # (seq_len, 1, d_model)
            x_wip = torch.cat([x_wip, x_pred_embed[-1:]], dim=0)

        # # with teacher forcing (same as above)
        # x_emb = self.embedding(x_i.unsqueeze(1)) * math.sqrt(self.d_model) # (seq_len, 1, d_model)
        # first_tokens = self.dummy_first_token.repeat(1, x_emb.shape[1], 1) # (1, 1, d_model)
        # x_tf_in = torch.cat([first_tokens, x_emb[:-1, :, :]], dim=0) # (seq_len, 1, d_model)

        # padding_mask = (x_i==self.pad_token_id).unsqueeze(0) # (bs, seq_len)
        # x_enc_proj = x_enc.repeat(x_emb.shape[0], x_emb.shape[1], 1) # (seq_len, bs, d_model)
        # y_hat = self.forward_dec(x_tf_in, x_enc_proj, padding_mask=padding_mask) # (seq_len, bs, n_tokens)
        # y_pred = torch.argmax(y_hat, dim=2).T # (bs, seq_len)
        # print(y_pred)




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




