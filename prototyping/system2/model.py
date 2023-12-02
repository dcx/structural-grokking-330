import torch, torch.nn as nn, torch.utils.data as data, torch.nn.functional as F
import lightning as L
import math
import torchmetrics


# Define entire system as a LightningModule
class PlanTransformer(L.LightningModule):
    """
    A LightningModule (nn.Module subclass) defines a full *system*
    (ie: an LLM, diffusion model, autoencoder, or simple image classifier).
    """

    def __init__(self, d_model, nhead, num_encoder_layers, dropout, ntoken, lr, pad_token_id, weight_decay):
        super().__init__()

        self.d_model = d_model
        self.lr = lr
        self.pad_token_id = pad_token_id
        self.wd = weight_decay
 
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.embedding = nn.Embedding(ntoken, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.linear = nn.Linear(d_model, ntoken)

        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=ntoken, ignore_index=pad_token_id)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=ntoken, ignore_index=pad_token_id)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor=None) -> torch.Tensor:
        # in lightning, forward defines the prediction/inference actions
        x = x.T # (seq_len, bs)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        x = self.linear(x)
        return x # (seq_len, bs, ntoken)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y, padding_mask = batch # (bs, seq_len)
        y_hat = self(x, padding_mask=padding_mask) # (seq_len, bs, ntoken)
        ce_y_hat = torch.permute(y_hat, (1, 2, 0)) # (bs, ntoken, seq_len)
        loss = F.cross_entropy(ce_y_hat, y, ignore_index=self.pad_token_id)
        self.log("train_loss", loss)

        # accuracy
        y_pred = torch.argmax(y_hat, dim=2).T # (bs, seq_len)
        self.train_acc(y_pred, y)
        self.log('train_acc_step', self.train_acc)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, padding_mask = batch # (seq_len, batch_size)
        y_hat = self(x, padding_mask=padding_mask) # (seq_len, bs, ntoken)
        ce_y_hat = torch.permute(y_hat, (1, 2, 0)) # (bs, ntoken, seq_len)
        loss = F.cross_entropy(ce_y_hat, y, ignore_index=self.pad_token_id)
        self.log("val_loss", loss)

        # accuracy
        y_pred = torch.argmax(y_hat, dim=2).T # (bs, seq_len)
        self.val_acc(y_pred, y)
        self.log('val_acc_step', self.val_acc)

        return loss

    def on_train_epoch_end(self):
        self.log('train_acc_epoch', self.train_acc.compute())
        self.train_acc.reset()

    def on_val_epoch_end(self):
        self.log('val_acc_epoch', self.val_acc.compute())
        self.val_acc.reset()



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer


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