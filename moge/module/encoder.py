from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from transformers import AlbertConfig

from .albert import AlbertModel


class AlbertEncoder(nn.Module):
    def __init__(self, config: AlbertConfig):
        super(AlbertEncoder, self).__init__()

        self.albert = AlbertModel(config)

    def forward(self, input_seqs):
        attention_mask = (input_seqs > 0).type(torch.int)

        outputs = self.albert(
            input_ids=input_seqs,
            attention_mask=attention_mask,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
        )
        last_hidden_state, pooled_output = outputs

        return pooled_output

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--encoding_dim', type=int, default=128)
        parser.add_argument('--vocab_size', type=int, default=22)
        parser.add_argument('--word_embedding_size', type=int, default=22)
        parser.add_argument('--max_length', type=int, default=1000)

        parser.add_argument('--num_hidden_layers', type=int, default=1)
        parser.add_argument('--num_hidden_groups', type=int, default=1)
        parser.add_argument('--num_attention_heads', type=int, default=4)
        parser.add_argument('--intermediate_size', type=int, default=256)

        return parser


class ConvLSTM(pl.LightningModule):
    def __init__(self, hparams):
        super(ConvLSTM, self).__init__()

        if not hasattr(hparams, "word_embedding_size") or hparams.word_embedding_size is None:
            hparams.word_embedding_size = hparams.vocab_size

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

    def init_hidden(self, batch_size):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn((2 if self.hparams.nb_lstm_bidirectional else 1),
                               batch_size, self.hparams.nb_lstm_units).type_as(self.fc_encoder.weight)
        hidden_b = torch.randn((2 if self.hparams.nb_lstm_bidirectional else 1),
                               batch_size, self.hparams.nb_lstm_units).type_as(self.fc_encoder.weight)
        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    def forward(self, input_seqs):
        batch_size, seq_len = input_seqs.size()
        X_lengths = (input_seqs > 0).sum(1)
        hidden = self.init_hidden(batch_size)

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
            X = self.conv1_dropout(X)
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
        _, hidden = self.lstm(X, hidden)  # (output, (h_n, c_n))

        X = hidden[0].permute(1, 0, 2)  # (batch_size, nb_layers, nb_lstm_units)
        X = X.reshape(batch_size, (
            2 if self.hparams.nb_lstm_bidirectional else 1) * self.hparams.nb_lstm_units)  # (batch_size, lstm_hidden)
        if self.hparams.nb_lstm_layernorm:
            X = self.lstm_layernorm(X)
        X = self.lstm_hidden_dropout(X)

        X = self.fc_encoder(X)
        return X

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--encoding_dim', type=int, default=128)
        parser.add_argument('--embedding_dim', type=int, default=256)
        parser.add_argument('--word_embedding_size', type=int, default=None)

        parser.add_argument('--nb_conv1_filters', type=int, default=192)
        parser.add_argument('--nb_conv1_kernel_size', type=int, default=10)
        parser.add_argument('--nb_conv1_dropout', type=float, default=0.2)
        parser.add_argument('--nb_conv1_batchnorm', type=bool, default=True)

        parser.add_argument('--nb_conv2_filters', type=int, default=128)
        parser.add_argument('--nb_conv2_kernel_size', type=int, default=3)
        parser.add_argument('--nb_conv2_batchnorm', type=bool, default=True)

        parser.add_argument('--nb_max_pool_size', type=int, default=13)

        parser.add_argument('--nb_lstm_units', type=int, default=100)
        parser.add_argument('--nb_lstm_bidirectional', type=bool, default=False)
        parser.add_argument('--nb_lstm_hidden_dropout', type=float, default=0.0)
        parser.add_argument('--nb_lstm_layernorm', type=bool, default=False)

        return parser
