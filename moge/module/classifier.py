from argparse import ArgumentParser

import torch
from torch import nn

import pytorch_lightning as pl
from transformers import AlbertConfig

from moge.module.embedder import GAT
from moge.module.encoder import ConvLSTM, AlbertEncoder


class EncoderEmbedderClassifier(pl.LightningModule):
    def __init__(self, hparams):
        super(EncoderEmbedderClassifier, self).__init__()

        if hparams.encoder == "ConvLSTM":
            self._encoder = ConvLSTM(hparams)
        elif hparams.encoder == "Albert":
            config = AlbertConfig(
                vocab_size=hparams.vocab_size,
                embedding_size=hparams.word_embedding_size,
                hidden_size=hparams.encoding_dim,
                num_hidden_layers=hparams.num_hidden_layers,
                num_hidden_groups=hparams.num_hidden_groups,
                hidden_dropout_prob=hparams.hidden_dropout_prob,
                attention_probs_dropout_prob=hparams.attention_probs_dropout_prob,
                num_attention_heads=hparams.num_attention_heads,
                intermediate_size=hparams.intermediate_size,
                type_vocab_size=1,
                max_position_embeddings=hparams.max_length,
            )
            self._encoder = AlbertEncoder(config)
        else:
            raise Exception("hparams.encoder must be one of {'ConvLSTM', 'Albert'}")

        if hparams.embedder == "GAT":
            self._embedder = GAT(hparams)
        else:
            raise Exception("hparams.embedder must be one of {'GAT'}")

        if hparams.classifier == "Dense":
            self._classifier = Dense(hparams)
        else:
            raise Exception("hparams.classifier must be one of {'Dense'}")

        self.hparams = hparams

        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, X):
        input_seqs, subnetwork = X["input_seqs"], X["subnetwork"]
        # print("input_seqs", input_seqs.shape)
        # print("subnetwork", len(subnetwork), [batch.shape for batch in subnetwork])
        # subnetwork = subnetwork[0].squeeze(0)
        encodings = self._encoder(input_seqs)
        embeddings = self._embedder(encodings, subnetwork)
        y_pred = self._classifier(embeddings)
        return y_pred

    def loss(self, Y_hat: torch.Tensor, Y, weights=None):
        Y = Y.type_as(Y_hat)
        idx = torch.nonzero(weights).view(-1)
        Y = Y[idx, :]
        Y_hat = Y_hat[idx, :]

        return self.criterion(Y_hat, Y)

    def get_encodings(self, X, batch_size=None, cuda=True):
        if not isinstance(X["input_seqs"], torch.Tensor):
            X = {k: torch.tensor(v).cuda() for k, v in X.items()}

        if cuda:
            X = {k: v.cuda() for k, v in X.items()}
        else:
            self._encoder.cpu()
            X = {k: v.cpu() for k, v in X.items()}

        if batch_size is not None:
            input_chunks = X["input_seqs"].split(split_size=batch_size, dim=0)
            encodings = []
            for i in range(len(input_chunks)):
                encodings.append(self._encoder.forward(input_chunks[i]))
            encodings = torch.cat(encodings, 0)
        else:
            encodings = self._encoder(X["input_seqs"])

        return encodings.detach().cpu().numpy()

    def get_embeddings(self, X, encodings=None, batch_size=None, cuda=True):
        """
        Get embeddings for a set of nodes in `X`.
        :param X: a dict with keys {"input_seqs", "subnetwork"}
        :param cuda (bool): whether to run computations in
        :return (np.array): a numpy array of size (node size, embedding dim)
        """
        if not isinstance(X["input_seqs"], torch.Tensor):
            X = {k: torch.tensor(v).cuda() for k, v in X.items()}

        if cuda:
            X = {k: v.cuda() for k, v in X.items()}
        else:
            X = {k: v.cpu() for k, v in X.items()}

        if encodings is None:
            encodings = self._encoder(X["input_seqs"])

        embeddings = self._embedder(encodings, X["subnetwork"])

        return embeddings.detach().cpu().numpy()

    def predict(self, embeddings, cuda=True):
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings)

        if cuda:
            embeddings = embeddings.cuda()
        else:
            embeddings = embeddings.cpu()

        y_pred = self._classifier(embeddings)
        return y_pred.detach().cpu().numpy()


class Dense(pl.LightningModule):
    def __init__(self, hparams) -> None:
        super(Dense, self).__init__()

        # Classifier
        self.fc_classifier = nn.Sequential(
            nn.Linear(hparams.embedding_dim, hparams.nb_cls_dense_size),
            nn.ReLU(),
            nn.Dropout(p=hparams.nb_cls_dropout),
            nn.Linear(hparams.nb_cls_dense_size, hparams.n_classes),
            # nn.Sigmoid()
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--nb_cls_dense_size', type=int, default=512)
        parser.add_argument('--nb_cls_dropout', type=float, default=0.2)
        parser.add_argument('--n_classes', type=int, default=128)
        return parser

    def forward(self, embeddings):
        return self.fc_classifier(embeddings)