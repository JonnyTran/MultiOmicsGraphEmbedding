from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping
from torch.autograd import Variable
from torch.nn import functional as F
from torch_geometric.nn import GATConv


class EncoderLSTM(nn.Module):
    def __init__(self, hparams):
        super(EncoderLSTM, self).__init__()
        self.hparams = hparams

        if self.hparams.word_embedding_size is None:
            self.hparams.word_embedding_size = self.hparams.vocab_size

        self.__build_model()

    def __build_model(self):
        # Encoder
        self.word_embedding = nn.Embedding(
            num_embeddings=self.hparams.vocab_size + 1,
            embedding_dim=self.hparams.word_embedding_size,
            padding_idx=0)
        self.conv1 = nn.Conv1d(
            in_channels=self.hparams.word_embedding_size,
            out_channels=self.hparams.nb_conv1d_filters,
            kernel_size=self.hparams.nb_conv1d_kernel_size,
        )
        self.conv_batchnorm = nn.BatchNorm1d(self.hparams.nb_conv1d_filters)
        self.conv1_dropout = nn.Dropout(p=self.hparams.nb_conv1d_dropout)

        self.lstm = nn.LSTM(
            input_size=self.hparams.nb_conv1d_filters,
            hidden_size=self.hparams.nb_lstm_units,
            bidirectional=self.hparams.nb_lstm_bidirectional,
            num_layers=1,
            batch_first=True, )
        self.lstm_layernorm = nn.LayerNorm(
            (2 if self.hparams.nb_lstm_bidirectional else 1) * self.hparams.nb_lstm_units * self.hparams.nb_lstm_layers)
        self.lstm_hidden_dropout = nn.Dropout(p=self.hparams.nb_lstm_hidden_dropout)
        self.fc_encoder = nn.Linear(
            (2 if self.hparams.nb_lstm_bidirectional else 1) * self.hparams.nb_lstm_units * self.hparams.nb_lstm_layers,
            self.hparams.encoding_dim)

        # Embedder
        self.embedder = GATConv(
            in_channels=self.hparams.encoding_dim,
            out_channels=int(self.hparams.embedding_dim / self.hparams.nb_attn_heads),
            heads=self.hparams.nb_attn_heads,
            concat=True,
            dropout=self.hparams.nb_attn_dropout
        )

        # Classifier
        self.fc_classifier = nn.Sequential(
            nn.Linear(self.hparams.embedding_dim, self.hparams.nb_cls_dense_size),
            nn.ReLU(),
            nn.Dropout(p=self.hparams.nb_cls_dropout),
            nn.Linear(self.hparams.nb_cls_dense_size, self.hparams.n_classes),
            nn.Sigmoid()
        )

    def init_hidden(self, batch_size):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn((2 if self.hparams.nb_lstm_bidirectional else 1) * self.hparams.nb_lstm_layers,
                               batch_size, self.hparams.nb_lstm_units).type_as(self.fc_encoder.weight)
        hidden_b = torch.randn((2 if self.hparams.nb_lstm_bidirectional else 1) * self.hparams.nb_lstm_layers,
                               batch_size, self.hparams.nb_lstm_units).type_as(self.fc_encoder.weight)
        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    def forward(self, X):
        input_seqs, subnetwork = X["input_seqs"], X["subnetwork"]
        # input_seqs, subnetwork = input_seqs.view(input_seqs.shape[1:]), subnetwork.view(subnetwork.shape[1:])

        encodings = self.get_encodings(input_seqs)
        # y_pred = F.sigmoid(encodings)
        embeddings = self.embedder(encodings, subnetwork)
        y_pred = self.fc_classifier(embeddings)
        return y_pred

    def get_encodings(self, input_seqs):
        batch_size, seq_len = input_seqs.size()
        X_lengths = (input_seqs > 0).sum(1)
        self.hidden = self.init_hidden(batch_size)

        X = self.word_embedding(input_seqs)
        X = X.permute(0, 2, 1)
        X = F.relu(F.max_pool1d(self.conv1(X), self.hparams.nb_max_pool_size))
        if self.hparams.nb_conv1d_batchnorm:
            X = self.conv_batchnorm(X)
        X = self.conv1_dropout(X)
        X = X.permute(0, 2, 1)

        X_lengths = (X_lengths - self.hparams.nb_conv1d_kernel_size + 1) / self.hparams.nb_max_pool_size
        X_lengths = torch.max(X_lengths, torch.ones_like(X_lengths))

        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True, enforce_sorted=False)
        _, self.hidden = self.lstm(X, self.hidden)

        X = self.hidden[0].permute(1, 0, 2)
        X = X.reshape(batch_size, (
            2 if self.hparams.nb_lstm_bidirectional else 1) * self.hparams.nb_lstm_layers * self.hparams.nb_lstm_units)

        if self.hparams.nb_lstm_layernorm:
            X = self.lstm_layernorm(X)
        X = self.lstm_hidden_dropout(X)
        X = self.fc_encoder(X)
        return X

    def loss(self, Y_hat, Y, weights=None):
        Y = Y.type_as(Y_hat)
        return F.binary_cross_entropy(Y_hat, Y, None)
        # return F.multilabel_soft_margin_loss(Y_hat, Y)

    def get_embeddings(self, X, cuda=True):
        if not isinstance(X["input_seqs"], torch.Tensor):
            X = {k: torch.tensor(v).cuda() for k, v in X.items()}

        if cuda:
            X = {k: v.cuda() for k, v in X.items()}
        else:
            X = {k: v.cpu() for k, v in X.items()}

        encodings = self.get_encodings(X["input_seqs"])
        embeddings = self.embedder(encodings, X["subnetwork"])

        return embeddings.detach().cpu().numpy()

    def predict(self, embeddings, cuda=True):
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings)

        if cuda:
            embeddings = embeddings.cuda()
        else:
            embeddings = embeddings.cpu()

        y_pred = self.fc_classifier(embeddings)
        return y_pred.detach().cpu().numpy()


def train(hparams):
    # init model
    model = EncoderLSTM(hparams)

    callbacks = [EarlyStopping(patience=10)]

    trainer = pl.Trainer(gpus=1,
                         #     distributed_backend='dp',
                         min_epochs=20,
                         max_epochs=50,
                         callbacks=callbacks,
                         weights_summary='top', )
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()

    # parametrize the network
    parser.add_argument('--encoding_dim', type=int, default=128)
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--n_classes', type=int, default=2000)
    parser.add_argument('--vocab_size', type=int, default=22)
    parser.add_argument('--word_embedding_size', type=int, default=None)

    parser.add_argument('--nb_conv1d_filters', type=int, default=192)
    parser.add_argument('--nb_conv1d_kernel_size', type=int, default=26)
    parser.add_argument('--nb_max_pool_size', type=int, default=13)
    parser.add_argument('--nb_conv1d_dropout', type=float, default=0.2)
    parser.add_argument('--nb_conv1d_layernorm', type=bool, default=True)

    parser.add_argument('--nb_lstm_layers', type=int, default=1)
    parser.add_argument('--nb_lstm_units', type=int, default=100)
    parser.add_argument('--nb_lstm_dropout', type=float, default=0.0)
    parser.add_argument('--nb_lstm_hidden_dropout', type=float, default=0.0)
    parser.add_argument('--nb_lstm_bidirectional', type=bool, default=False)
    parser.add_argument('--nb_lstm_layernorm', type=bool, default=False)

    parser.add_argument('--nb_attn_heads', type=int, default=4)
    parser.add_argument('--nb_attn_dropout', type=float, default=0.5)

    parser.add_argument('--nb_cls_dense_size', type=int, default=512)
    parser.add_argument('--nb_cls_dropout', type=float, default=0.2)

    parser.add_argument('--nb_weight_decay', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-3)

    # add all the available options to the trainer
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    train(args)
