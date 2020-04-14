import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class EncoderLSTM(nn.Module):
    def __init__(self, encoding_dim=128, vocab=None,
                 nb_lstm_layers=1, nb_lstm_units=100,
                 nb_conv1d_filters=192, nb_conv1d_kernel_size=26, nb_max_pool_size=2):
        super(EncoderLSTM, self).__init__()

        self.nb_lstm_layers = nb_lstm_layers
        self.nb_lstm_units = nb_lstm_units

        self.nb_conv1d_filters = nb_conv1d_filters
        self.nb_conv1d_kernel_size = nb_conv1d_kernel_size
        self.nb_max_pool_size = nb_max_pool_size

        self.vocab = vocab
        self.encoding_size = encoding_dim

        self.word_embedding = nn.Embedding(
            num_embeddings=len(self.vocab) + 1,
            embedding_dim=len(self.vocab),
            padding_idx=0)

        self.conv1 = nn.Conv1d(
            in_channels=len(self.vocab),
            out_channels=self.nb_conv1d_filters,
            kernel_size=self.nb_conv1d_kernel_size)

        self.lstm = nn.LSTM(
            input_size=self.nb_conv1d_filters,
            hidden_size=self.nb_lstm_units,
            num_layers=self.nb_lstm_layers,
            batch_first=True, )

        self.hidden_to_tag = nn.Linear(self.nb_lstm_units, self.encoding_size)

    def init_hidden(self, batch_size):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn(self.nb_lstm_layers, batch_size, self.nb_lstm_units).cuda()
        hidden_b = torch.randn(self.nb_lstm_layers, batch_size, self.nb_lstm_units).cuda()

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    def forward(self, X_input):
        batch_size, seq_len = X_input["input_seqs"].size()
        X_lengths = (X_input["input_seqs"] > 0).sum(1)

        self.hidden = self.init_hidden(batch_size)

        X = self.word_embedding(X_input["input_seqs"])
        X = X.permute(0, 2, 1)
        X = F.relu(F.max_pool1d(self.conv1(X), self.nb_max_pool_size))

        X = X.permute(0, 2, 1)
        X_lengths = (X_lengths - self.nb_conv1d_kernel_size) / self.nb_max_pool_size + 1
        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True, enforce_sorted=False)

        _, self.hidden = self.lstm(X, self.hidden)

        X = self.hidden[0].view(self.nb_lstm_layers * batch_size, self.nb_lstm_units)

        X = self.hidden_to_tag(X)
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
        # nb_tokens = int(torch.sum(mask).data[0])
        #
        # # pick the values for the label and zero out the rest with the mask
        # Y_hat = Y_hat[range(Y_hat.shape[0]), Y] * mask
        #
        # # compute cross entropy loss which ignores all <PAD> tokens
        # ce_loss = -torch.sum(Y_hat) / nb_tokens
        #
        # return ce_loss
