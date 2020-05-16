from argparse import ArgumentParser

from torch import nn


class Dense(nn.Module):
    def __init__(self, hparams) -> None:
        super(Dense, self).__init__()

        # Classifier
        self.fc_classifier = nn.Sequential(
            nn.Linear(hparams.embedding_dim, hparams.nb_cls_dense_size),
            nn.ReLU(),
            nn.Dropout(p=hparams.nb_cls_dropout),
            nn.Linear(hparams.nb_cls_dense_size, hparams.n_classes),
        )
        if "LOGITS" in hparams.loss_type or "FOCAL" in hparams.loss_type:
            print("INFO: Output of `_classifier` is logits")
        elif "SOFTMAX_CROSS_ENTROPY" in hparams.loss_type:
            self.fc_classifier.add_module("pred_activation", nn.Softmax())
        else:
            self.fc_classifier.add_module("pred_activation", nn.Sigmoid())

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--nb_cls_dense_size', type=int, default=512)
        parser.add_argument('--nb_cls_dropout', type=float, default=0.2)
        parser.add_argument('--n_classes', type=int, default=128)
        return parser

    def forward(self, embeddings):
        return self.fc_classifier(embeddings)


