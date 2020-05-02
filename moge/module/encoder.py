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

        if hparams.word_embedding_size is None:
            hparams.word_embedding_size = hparams.vocab_size

        if hasattr(hparams, "class_weights"):
            hparams.class_weights

        self.hparams = hparams

        # Encoder
        self.word_embedding = nn.Embedding(
            num_embeddings=hparams.vocab_size + 1,
            embedding_dim=hparams.word_embedding_size,
            padding_idx=0)
        self.conv1 = nn.Conv1d(
            in_channels=hparams.word_embedding_size,
            kernel_size=hparams.nb_conv1_kernel_size,
            out_channels=hparams.nb_conv1_filters,
        )
        self.conv1_batchnorm = nn.BatchNorm1d(hparams.nb_conv1_filters)
        self.conv1_dropout = nn.Dropout(p=hparams.nb_conv1_dropout)

        self.conv2 = nn.Conv1d(
            in_channels=hparams.nb_conv1_filters,
            kernel_size=hparams.nb_conv2_kernel_size,
            out_channels=hparams.nb_conv2_filters,
        )
        self.conv2_batchnorm = nn.BatchNorm1d(hparams.nb_conv2_filters)

        self.lstm = nn.LSTM(
            num_layers=1,
            input_size=hparams.nb_conv2_filters if hparams.nb_conv2_kernel_size > 1 else hparams.nb_conv1_filters,
            hidden_size=hparams.nb_lstm_units,
            bidirectional=hparams.nb_lstm_bidirectional,
            batch_first=True, )
        self.lstm_layernorm = nn.LayerNorm((2 if hparams.nb_lstm_bidirectional else 1) * hparams.nb_lstm_units)
        self.lstm_hidden_dropout = nn.Dropout(p=hparams.nb_lstm_hidden_dropout)
        self.fc_encoder = nn.Linear(
            (2 if hparams.nb_lstm_bidirectional else 1) * hparams.nb_lstm_units, hparams.encoding_dim)

        # Embedder
        self.embedder = GATConv(
            in_channels=hparams.encoding_dim,
            out_channels=int(hparams.embedding_dim / hparams.nb_attn_heads),
            heads=hparams.nb_attn_heads,
            concat=True,
            dropout=hparams.nb_attn_dropout
        )

        # Classifier
        self.fc_classifier = nn.Sequential(
            nn.Linear(hparams.embedding_dim, hparams.nb_cls_dense_size),
            nn.ReLU(),
            nn.Dropout(p=hparams.nb_cls_dropout),
            nn.Linear(hparams.nb_cls_dense_size, hparams.n_classes),
            nn.Sigmoid()
        )

    def init_hidden(self, batch_size):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn((2 if self.hparams.nb_lstm_bidirectional else 1),
                               batch_size, self.hparams.nb_lstm_units).type_as(self.fc_encoder.weight)
        hidden_b = torch.randn((2 if self.hparams.nb_lstm_bidirectional else 1),
                               batch_size, self.hparams.nb_lstm_units).type_as(self.fc_encoder.weight)
        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    def forward(self, X):
        input_seqs, subnetwork = X["input_seqs"], X["subnetwork"]
        # input_seqs, subnetwork = input_seqs.view(input_seqs.shape[1:]), subnetwork.view(subnetwork.shape[1:])

        encodings = self.get_encodings(input_seqs)
        y_pred = F.sigmoid(encodings)
        # embeddings = self.embedder(encodings, subnetwork)
        # y_pred = self.fc_classifier(embeddings)
        return y_pred

    def get_encodings(self, input_seqs):
        batch_size, seq_len = input_seqs.size()
        X_lengths = (input_seqs > 0).sum(1)
        self.hidden = self.init_hidden(batch_size)

        X = self.word_embedding(input_seqs)

        # Conv_1
        X = X.permute(0, 2, 1)  # (batch_size, n_channels, X_lengths)
        X = F.relu(self.conv1(X))
        if self.hparams.nb_conv1_batchnorm:
            X = self.conv1_batchnorm(X)
        X = self.conv1_dropout(X)

        X_lengths = (X_lengths - self.hparams.nb_conv1_kernel_size + 1)

        # Conv_2
        if self.hparams.nb_conv2_kernel_size > 1:
            X = F.relu(self.conv2(X))
            if self.hparams.nb_conv2_batchnorm:
                X = self.conv2_batchnorm(X)
            X_lengths = (X_lengths - self.hparams.nb_conv2_kernel_size + 1)

        # Maxpool
        X = F.max_pool1d(X, self.hparams.nb_max_pool_size)
        X = X.permute(0, 2, 1)  # {}
        X_lengths = X_lengths / self.hparams.nb_max_pool_size
        X_lengths = torch.max(X_lengths, torch.ones_like(X_lengths))

        # LSTM
        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True, enforce_sorted=False)
        _, self.hidden = self.lstm(X, self.hidden)  # (output, (h_n, c_n))

        X = self.hidden[0].permute(1, 0, 2)  # (batch_size, nb_layers, nb_lstm_units)
        X = X.reshape(batch_size, (
            2 if self.hparams.nb_lstm_bidirectional else 1) * self.hparams.nb_lstm_units)  # (batch_size, lstm_hidden)
        if self.hparams.nb_lstm_layernorm:
            X = self.lstm_layernorm(X)
        X = self.lstm_hidden_dropout(X)

        X = self.fc_encoder(X)
        return X

    def loss(self, Y_hat, Y, weights=None):
        Y = Y.type_as(Y_hat)
        idx = torch.nonzero(weights).view(-1)
        Y = Y[idx]
        Y_hat = Y_hat[idx]
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
