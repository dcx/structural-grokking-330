import torch, torch.nn as nn, torch.utils.data as data, torch.nn.functional as F
import lightning as L
import math
import torchmetrics
import random

class System1(nn.Module):
    def __init__(self, d_model, n_enc_heads, n_enc_layers, n_unique_tokens, pad_token_id, dropout):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.embedding = nn.Embedding(n_unique_tokens, d_model, padding_idx=pad_token_id)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_enc_heads, dropout=dropout, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_enc_layers)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        #self.linear.bias.data.zero_()
        #self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, x_raw, wm, padding_mask):
        """
        Single System 1 step.
        Looks at input and current state, proposes a new state.
        - x_raw: (bs, seq_len)
        - wm: (seq_len, bs, d_model)
        - padding_mask: (bs, seq_len)

        Returns s1's view of x and proposed wm, concatenated: 
        - (seq_len, bs, d_model*2)
        """
        # embed x_raw
        x = x_raw.T # (seq_len, bs)
        x = self.embedding(x) * math.sqrt(self.d_model) # (seq_len, bs, d_model)

        # concat wm,x and add positional encoding
        enc_in = torch.cat([wm, x], dim=0) # (4+seq_len, bs, d_model)
        enc_in = self.pos_encoder(enc_in) # (4+seq_len, bs, d_model)

        # double up padding mask
        padding_mask = torch.cat([torch.zeros((x_raw.shape[0], 4), device=x_raw.device).bool(), padding_mask], dim=1) # (bs, 4+seq_len)

        # push through transformer
        s1_proposed = self.transformer_encoder(enc_in, src_key_padding_mask=padding_mask) # (4+seq_len, bs, d_model)

        return s1_proposed

class System2(nn.Module):
    def __init__(self, d_model, n_enc_heads, n_enc_layers, pad_token_id, dropout):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_enc_heads, dropout=dropout, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_enc_layers)
        #self.linear = nn.Linear(d_model, d_model)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        #self.linear.bias.data.zero_()
        #self.linear.weight.data.uniform_(-initrange, initrange)


    def forward(self, s1_proposed, wm_current, padding_mask):
        """
        Single System 2 step.
        Looks at current and proposed state, generates next state.
        - s1_proposed (seq_len*2, bs, d_model)
        - wm_current (seq_len, bs, d_model)

        Returns wm_next:
        - (seq_len, bs, d_model)
        """

        # concat s1_proposal and wm_current, push through transformer
        enc_in  = torch.cat([s1_proposed, wm_current], dim=0) # (seq_len*3, bs, d_model)
        padding_mask = torch.cat([padding_mask, padding_mask, padding_mask], dim=1) # (bs, seq_len*3)
        enc_out = self.transformer_encoder(enc_in, src_key_padding_mask=padding_mask) # (seq_len*3, bs, d_model)

        # next state: take middle third of output (x, wm_proposed, wm_current)
        wm_next = enc_out[enc_out.shape[0]//3:2*enc_out.shape[0]//3, :, :] # (seq_len, bs, d_model)
        # wm_next = self.linear(enc_out) # (seq_len, bs, d_model)
        return wm_next

class S1OutputTranslator(nn.Module):
    def __init__(self, d_model, n_enc_heads, n_enc_layers, pad_token_id, n_output_tokens, dropout):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id

        #encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_enc_heads, dropout=dropout, activation='gelu')
        #self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_enc_layers)
        self.linear = nn.Linear(d_model, n_output_tokens)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, s1_proposed, padding_mask):
        """
        Future: This gets run every timestep with a flag to indicate whether
        it's a per-step, intermediate, or final prediction.
        That way it can only compute the relevant part of the output.
        To start: Only gets run on final prediction.

        Output prediction: Takes s1_proposed and returns a prediction of the
        final output.

        - s1_proposed (seq_len+4, bs, d_model)
        """

        # double up padding mask
        # padding_mask = torch.cat([padding_mask, padding_mask], dim=1) # (bs, seq_len*2)

        # push through transformer
        # enc_out = self.transformer_encoder(s1_proposed, src_key_padding_mask=padding_mask) # (seq_len*2, bs, d_model)

        # output pred: reduce to n_output_tokens over x
        # enc_out_x = enc_out[:enc_out.shape[0]//2, :, :] # (seq_len, bs, d_model)
        output_pred = self.linear(s1_proposed[4:, :, :]) # (seq_len, bs, n_output_tokens)

        return output_pred


        #confidence = torch.tensor(0.5) # TODO: this should be a function of wm
        #x_pred = wm # TODO: shape should be same as x_raw
        # timestep prediction goes here
        #return confidence, x_pred # new predictions go here

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# Define entire system as a LightningModule
class BPTTTransformer(L.LightningModule):
    """
    A LightningModule (nn.Module subclass) defines a full *system*
    (ie: an LLM, diffusion model, autoencoder, or simple image classifier).
    """

    def __init__(self, d_model, n_enc_heads, n_enc_layers, dropout, n_unique_tokens, n_output_tokens, lr, weight_decay, pad_token_id, max_steps):
        super().__init__()

        self.d_model = d_model
        self.lr = lr
        self.wd = weight_decay
        self.pad_token_id = pad_token_id
        self.max_steps = max_steps

        self.s1 = System1(d_model, n_enc_heads, n_enc_layers, n_unique_tokens, pad_token_id, dropout)
        self.s2 = System2(d_model, n_enc_heads, n_enc_layers, pad_token_id, dropout)
        self.s1o = S1OutputTranslator(d_model, n_enc_heads, n_enc_layers, pad_token_id, n_output_tokens, dropout)

        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_unique_tokens, ignore_index=pad_token_id)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_unique_tokens, ignore_index=pad_token_id)
        self.train_rowwise_acc = torchmetrics.classification.Accuracy(task="binary", num_classes=2)
        self.val_rowwise_acc = torchmetrics.classification.Accuracy(task="binary", num_classes=2)

        self.pos_encoder = PositionalEncoding(2*d_model, dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor=None, n_steps: int=1) -> torch.Tensor:
        """
        x: (bs, seq_len)
        """

        # initialize wm with zeros
        wm_current = torch.zeros((4, x.shape[0], self.d_model)).to(x.device) # (4, bs, d_model)
        # wm_current = torch.zeros((x.shape[1], x.shape[0], self.d_model)).to(x.device) # (seq_len, bs, d_model)

        for i in range(n_steps):
            # run s1
            s1_proposed = self.s1(x, wm_current, padding_mask) # (4+seq_len, bs, d_model)
            # run s2 and update wm
            # wm_current = self.s2(s1_proposed, wm_current, padding_mask) # (seq_len, bs, d_model)

        # run s2o on final step
        output_pred = self.s1o(s1_proposed, padding_mask) # (seq_len, bs, n_output_tokens)
        return output_pred # (seq_len, bs, ntoken)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y, padding_mask = batch # (bs, seq_len)

        # pick a random number of steps, from 2 to max_steps
        # n_steps = 1 # random.randint(2, self.max_steps)
        # x is first step, y is n_steps later
        #x = b[:, 0, :] # (bs, seq_len)
        #y = b[:, n_steps, :] # (bs, seq_len)
        #padding_mask = padding_mask[:, 0, :] # (bs, seq_len)

        # run model for n_steps
        y_hat = self(x, padding_mask=padding_mask, n_steps=1) # (seq_len, bs, n_output_tokens)
        ce_y_hat = torch.permute(y_hat, (1, 2, 0)) # (bs, n_output_tokens, seq_len)
        loss = F.cross_entropy(ce_y_hat, y, ignore_index=self.pad_token_id)
        
        self.log("train_loss", loss)

        # accuracy
        y_pred = torch.argmax(y_hat, dim=2).T # (bs, seq_len)
        self.train_acc(y_pred, y)
        self.log('train_acc_step', self.train_acc)

        # rowwise accuracy: do we get the whole row right?
        y_match = (y_pred == y) | (y == self.pad_token_id)
        y_match_row = y_match.all(dim=1).long() # (bs,)
        self.train_rowwise_acc(y_match_row, torch.ones_like(y_match_row))
        self.log('train_rowwise_acc_step', self.train_rowwise_acc)

        return loss
    
    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y, padding_mask = batch # (bs, seq_len)

        # pick a random number of steps, from 2 to max_steps
        #n_steps = random.randint(2, self.max_steps)
        # x is first step, y is n_steps later
        #x = b[:, 0, :] # (bs, seq_len)
        #y = b[:, n_steps, :] # (bs, seq_len)
        #padding_mask = padding_mask[:, 0, :] # (bs, seq_len)

        # run model for n_steps
        y_hat = self(x, padding_mask=padding_mask, n_steps=1) # (seq_len, bs, n_output_tokens)
        ce_y_hat = torch.permute(y_hat, (1, 2, 0)) # (bs, n_output_tokens, seq_len)
        loss = F.cross_entropy(ce_y_hat, y, ignore_index=self.pad_token_id)
        
        self.log("val_loss", loss)

        # accuracy
        y_pred = torch.argmax(y_hat, dim=2).T # (bs, seq_len)
        self.val_acc(y_pred, y)
        self.log('val_acc_step', self.train_acc)

        # rowwise accuracy: do we get the whole row right?
        y_match = (y_pred == y) | (y == self.pad_token_id)
        y_match_row = y_match.all(dim=1).long() # (bs,)
        self.val_rowwise_acc(y_match_row, torch.ones_like(y_match_row))
        self.log('val_rowwise_acc_step', self.val_rowwise_acc)

        return loss


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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer





if __name__ == "__main__":
    
    # shake out end-to-end
    d_model = 64
    n_enc_heads = 2
    n_enc_layers = 2
    n_unique_tokens = 5 # (0,1,ca_rule_0,ca_rule_1,pad)
    n_output_tokens = 2 # (0,1)
    pad_token_id = 4
    dropout = 0.1

    s1 = System1(d_model, n_enc_heads, n_enc_layers, n_unique_tokens, pad_token_id, dropout)
    s2 = System2(d_model, n_enc_heads, n_enc_layers, pad_token_id, dropout)
    s2o = S1OutputTranslator(d_model, n_enc_heads, n_enc_layers, pad_token_id, n_output_tokens, dropout)

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

