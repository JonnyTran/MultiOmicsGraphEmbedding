from argparse import ArgumentParser

import torch
from torch import nn

import pytorch_lightning as pl


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
