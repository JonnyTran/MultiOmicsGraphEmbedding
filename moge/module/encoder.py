import numpy as np

import pytorch_lightning as pl
import torch
import torch.nn as nn
from ignite.metrics import Precision, Recall
from torch.autograd import Variable
from torch.nn import functional as F
from torch_geometric.nn import GATConv
from argparse import ArgumentParser


class EncoderLSTM(pl.LightningModule):
    def __init__(self, hparams):
        super(EncoderLSTM, self).__init__()
        self.hparams = hparams

        if self.hparams.word_embedding_size is None:
            self.hparams.word_embedding_size = self.hparams.vocab_size

        self.__build_model()
        self.init_metrics()

    def __build_model(self):
        # Encoder
        self.word_embedding = nn.Embedding(
            num_embeddings=self.hparams.vocab_size + 1,
            embedding_dim=self.hparams.word_embedding_size,
            padding_idx=0)
        self.conv1 = nn.Conv1d(
            in_channels=self.hparams.word_embedding_size,
            out_channels=self.hparams.nb_conv1d_filters,
            kernel_size=self.hparams.nb_conv1d_kernel_size)
        self.conv1_dropout = nn.Dropout(p=self.hparams.nb_conv1d_dropout)

        self.lstm = nn.LSTM(
            input_size=self.hparams.nb_conv1d_filters,
            hidden_size=self.hparams.nb_lstm_units,
            num_layers=self.hparams.nb_lstm_layers,
            dropout=self.hparams.nb_lstm_dropout,
            bidirectional=self.hparams.nb_lstm_bidirectional,
            batch_first=True, )
        self.lstm_hidden_dropout = nn.Dropout(p=self.hparams.nb_lstm_hidden_dropout)
        self.lstm_layernorm = nn.LayerNorm(self.hparams.nb_lstm_units * self.hparams.nb_lstm_layers)
        self.fc_encoder = nn.Linear(
            self.hparams.nb_lstm_units * self.hparams.nb_lstm_layers * (2 if self.hparams.nb_lstm_bidirectional else 1),
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
                               batch_size, self.hparams.nb_lstm_units).type_as(
            self.fc_encoder.weight)
        hidden_b = torch.randn((2 if self.hparams.nb_lstm_bidirectional else 1) * self.hparams.nb_lstm_layers,
                               batch_size, self.hparams.nb_lstm_units).type_as(
            self.fc_encoder.weight)

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    def forward(self, input_seqs, subnetwork):
        encodings = self.get_encodings(input_seqs)
        X = F.sigmoid(encodings)
        # Embedder
        # X = self.embedder(encodings, subnetwork)
        # Classifier
        # X = self.fc_classifier(X)
        return X

    def get_encodings(self, input_seqs):
        batch_size, seq_len = input_seqs.size()
        X_lengths = (input_seqs > 0).sum(1)
        self.hidden = self.init_hidden(batch_size)
        X = self.word_embedding(input_seqs)
        X = X.permute(0, 2, 1)
        X = F.relu(F.max_pool1d(self.conv1(X), self.hparams.nb_max_pool_size))
        X = self.conv1_dropout(X)
        if self.hparams.nb_conv1d_layernorm:
            X = F.layer_norm(X, X.shape[1:])

        X = X.permute(0, 2, 1)
        X_lengths = (X_lengths - self.hparams.nb_conv1d_kernel_size) / self.hparams.nb_max_pool_size

        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True, enforce_sorted=False)
        _, self.hidden = self.lstm(X, self.hidden)

        X = self.hidden[0].permute(1, 0, 2)
        X = X.reshape(batch_size, (
            2 if self.hparams.nb_lstm_bidirectional else 1) * self.hparams.nb_lstm_layers * self.hparams.nb_lstm_units)

        X = self.lstm_hidden_dropout(X)
        if self.hparams.nb_lstm_layernorm:
            X = self.lstm_layernorm(X)
        X = self.fc_encoder(X)
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

    def training_step(self, batch, batch_nb):
        X, y, train_weights = batch
        input_seqs, subnetwork = X["input_seqs"], X["subnetwork"]

        # input_seqs, subnetwork = input_seqs.view(input_seqs.shape[1:]), subnetwork.view(subnetwork.shape[1:])
        # y = y.view(y.shape[1:])

        Y_hat = self.forward(input_seqs, subnetwork)
        loss = self.loss(Y_hat, y, None)

        self.update_metrics(Y_hat, y, training=True)
        progress_bar = {
            "precision": self.precision.compute(),
            "recall": self.recall.compute()
        }

        return {"loss": loss,
                'progress_bar': progress_bar,
                }

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        tensorboard_logs = {
            "loss": avg_loss,
            "precision": self.precision.compute(),
            "recall": self.recall.compute(),
        }
        self.reset_metrics(training=True)
        return {"loss": avg_loss,
                "progress_bar": tensorboard_logs,
                "log": tensorboard_logs,
                }

    def validation_step(self, batch, batch_nb):
        X, y, train_weights = batch
        input_seqs, subnetwork = X["input_seqs"], X["subnetwork"]

        # input_seqs, subnetwork = input_seqs.view(input_seqs.shape[1:]), subnetwork.view(subnetwork.shape[1:])
        # y = y.view(y.shape[1:])

        Y_hat = self.forward(input_seqs, subnetwork)
        loss = self.loss(Y_hat, y, None)

        self.update_metrics(Y_hat, y, training=False)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {
            "val_loss": avg_loss,
            "val_precision": self.precision_val.compute(),
            "val_recall": self.recall_val.compute(),
        }

        results = {"val_loss": avg_loss,
                   "progress_bar": tensorboard_logs,
                   "log": tensorboard_logs}
        self.reset_metrics(training=False)

        return results

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

    def init_metrics(self):
        self.precision = Precision(average=True, is_multilabel=True)
        self.recall = Recall(average=True, is_multilabel=True)
        self.precision_val = Precision(average=True, is_multilabel=True)
        self.recall_val = Recall(average=True, is_multilabel=True)

    def update_metrics(self, y_pred, y_true, training):
        if training:
            self.precision.update(((y_pred > 0.5).type_as(y_true), y_true))
            self.recall.update(((y_pred > 0.5).type_as(y_true), y_true))
        else:
            self.precision_val.update(((y_pred > 0.5).type_as(y_true), y_true))
            self.recall_val.update(((y_pred > 0.5).type_as(y_true), y_true))

    def reset_metrics(self, training):
        if training:
            self.precision.reset()
            self.recall.reset()
        else:
            self.precision_val.reset()
            self.recall_val.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.lr,
                                     # weight_decay=self.hparams.nb_weight_decay
                                     )
        return optimizer


def main(hparams):
    # init model
    model = EncoderLSTM(hparams)

    trainer = pl.Trainer()
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()

    # parametrize the network
    parser.add_argument('--encoding_dim', type=int, default=128)
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--n_classes', type=int, default=2000)
    parser.add_argument('--vocab', type=int, default=22)
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
    main(args)
