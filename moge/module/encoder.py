import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class EncoderLSTM(nn.Module):
    def __init__(self, encoding_dim=128, vocab=None,
                 nb_lstm_layers=1, nb_lstm_units=100, nb_lstm_dropout=0.2,
                 nb_conv1d_filters=192, nb_conv1d_kernel_size=26, nb_max_pool_size=2, nb_conv1d_dropout=0.2,
                 nb_conv1d_batchnorm=True):
        super(EncoderLSTM, self).__init__()
        self.vocab = vocab
        self.word_embedding_size = len(self.vocab)

        self.nb_conv1d_filters = nb_conv1d_filters
        self.nb_conv1d_kernel_size = nb_conv1d_kernel_size
        self.nb_max_pool_size = nb_max_pool_size
        self.nb_conv1d_dropout = nb_conv1d_dropout
        self.nb_conv1d_batchnorm = nb_conv1d_batchnorm

        self.nb_lstm_layers = nb_lstm_layers
        self.nb_lstm_units = nb_lstm_units
        self.nb_lstm_dropout = nb_lstm_dropout

        self.encoding_size = encoding_dim

        self.word_embedding = nn.Embedding(
            num_embeddings=len(self.vocab) + 1,
            embedding_dim=self.word_embedding_size,
            padding_idx=0)

        self.conv1 = nn.Conv1d(
            in_channels=self.word_embedding_size,
            out_channels=self.nb_conv1d_filters,
            kernel_size=self.nb_conv1d_kernel_size)

        self.conv1_dropout = nn.Dropout(p=self.nb_conv1d_dropout)

        self.lstm = nn.LSTM(
            input_size=self.nb_conv1d_filters,
            hidden_size=self.nb_lstm_units,
            num_layers=self.nb_lstm_layers,
            dropout=self.nb_lstm_dropout,
            batch_first=True, )

        self.batchnorm = nn.BatchNorm1d(num_features=self.nb_lstm_units)

        self.encoder = nn.Linear(self.nb_lstm_units, self.encoding_size)

    def init_hidden(self, batch_size):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.zeros(self.nb_lstm_layers, batch_size, self.nb_lstm_units).cuda()
        hidden_b = torch.zeros(self.nb_lstm_layers, batch_size, self.nb_lstm_units).cuda()

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    def forward(self, input_seqs):
        batch_size, seq_len = input_seqs.size()
        X_lengths = (input_seqs > 0).sum(1)

        self.hidden = self.init_hidden(batch_size)

        X = self.word_embedding(input_seqs)
        X = X.permute(0, 2, 1)
        X = F.relu(F.max_pool1d(self.conv1(X), self.nb_max_pool_size))
        X = self.conv1_dropout(X)

        X = X.permute(0, 2, 1)
        X_lengths = (X_lengths - self.nb_conv1d_kernel_size) / self.nb_max_pool_size + 1
        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True, enforce_sorted=False)

        _, self.hidden = self.lstm(X, self.hidden)

        X = self.hidden[0].view(self.nb_lstm_layers * batch_size, self.nb_lstm_units)

        if self.nb_conv1d_batchnorm:
            X = self.batchnorm(X)

        X = self.encoder(X)

        X = F.sigmoid(X)

        return X

    def loss(self, Y_hat, Y, weights=None):
        Y = Y.type_as(Y_hat)
        return F.binary_cross_entropy(Y_hat, Y, weights, reduction="mean")

        # Y = Y.view(-1)
        #
        # # flatten all predictions
        # Y_hat = Y_hat.view(-1, self.n_classes)
        #
        # # create a mask by filtering out all tokens that ARE NOT the padding token
        # tag_pad_token = 0
        # mask = (Y > tag_pad_token).float()
        #
        # # count how many tokens we have
        #
        # nb_tokens = int(torch.sum(mask).data[0])
        # # pick the values for the label and zero out the rest with the mask
        # Y_hat = Y_hat[range(Y_hat.shape[0]), Y] * mask
        #
        # # compute cross entropy loss which ignores all <PAD> tokens
        # ce_loss = -torch.sum(Y_hat) / nb_tokens
        #
        # return ce_loss
