import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class EncoderLSTM(nn.Module):
    def __init__(self, nb_layers, nb_lstm_units=100, embedding_dim=128, batch_size=100, vocab=None, n_classes=None):
        super(EncoderLSTM, self).__init__()

        self.nb_lstm_layers = nb_layers
        self.nb_lstm_units = nb_lstm_units

        self.batch_size = batch_size
        self.vocab = vocab
        self.n_classes = n_classes

        self.nb_max_pool_size = 2
        self.nb_conv1d_filters = 192
        self.nb_conv1d_kernel_size = 5

        self.embedding_dim = embedding_dim

        self.word_embedding = nn.Embedding(
            num_embeddings=len(self.vocab) + 1,
            embedding_dim=len(self.vocab),
            padding_idx=0)

        self.conv1 = nn.Conv1d(
            len(self.vocab),
            self.nb_conv1d_filters,
            kernel_size=self.nb_conv1d_kernel_size)

        self.lstm = nn.LSTM(
            input_size=self.nb_conv1d_kernel_size,
            hidden_size=self.nb_lstm_units,
            num_layers=self.nb_lstm_layers,
            batch_first=True, )

        self.hidden_to_tag = nn.Linear(self.nb_lstm_units, self.n_classes)

    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn(self.nb_lstm_layers, self.batch_size, self.nb_lstm_units)
        hidden_b = torch.randn(self.nb_lstm_layers, self.batch_size, self.nb_lstm_units)

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    def forward(self, X):
        self.hidden = self.init_hidden()
        X_lengths = (X > 0).sum(1)

        X = self.word_embedding(X)
        X = X.permute(0, 2, 1)
        X = F.relu(F.max_pool1d(self.conv1(X), self.nb_max_pool_size))

        X = X.permute(0, 2, 1)
        X_lengths = (X_lengths - self.nb_conv1d_kernel_size) / self.nb_max_pool_size
        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True, enforce_sorted=False)

        X, self.hidden = self.lstm(X, self.hidden)

        return self.hidden[0]
        # X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        # run through actual linear layer
        # X = self.hidden_to_tag(X)
        # X = F.log_softmax(X, dim=1)
        # Y_hat = X.view(batch_size, self.n_classes)
        # return Y_hat

    def loss(self, Y_hat, Y):
        Y = Y.view(-1)

        # flatten all predictions
        Y_hat = Y_hat.view(-1, self.n_classes)

        # create a mask by filtering out all tokens that ARE NOT the padding token
        tag_pad_token = 0
        mask = (Y > tag_pad_token).float()

        # count how many tokens we have
        nb_tokens = int(torch.sum(mask).data[0])

        # pick the values for the label and zero out the rest with the mask
        Y_hat = Y_hat[range(Y_hat.shape[0]), Y] * mask

        # compute cross entropy loss which ignores all <PAD> tokens
        ce_loss = -torch.sum(Y_hat) / nb_tokens

        return ce_loss
