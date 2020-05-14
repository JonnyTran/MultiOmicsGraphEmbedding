import numpy as np
import torch
from torch import nn
from transformers import AlbertConfig

from moge.module.classifier import Dense
from moge.module.embedder import GAT
from moge.module.encoder import ConvLSTM, AlbertEncoder
from moge.module.losses import ClassificationLoss


class MultiplexConcatEmbedder(nn.Module):
    def __init__(self, hparams):
        super(MultiplexConcatEmbedder, self).__init__()

        assert isinstance(hparams.encoder, dict)
        assert isinstance(hparams.embedder, dict)
        # assert isinstance(hparams.vocab_size, dict)
        self.hparams = hparams

        self._encoder = {}
        for seq_type, encoder in hparams.encoder.items():
            if encoder == "ConvLSTM":
                self.__setattr__("_encoder_" + seq_type, ConvLSTM(hparams))
                self._encoder[seq_type] = self.__getattr__("_encoder_" + seq_type)
            elif encoder == "Albert":
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
                self._encoder[seq_type] = AlbertEncoder(config)
            else:
                raise Exception("hparams.encoder must be one of {'ConvLSTM', 'Albert'}")

        self._embedder = {}
        for subnetwork_type, embedder in hparams.embedder.items():
            if embedder == "GAT":
                self.__setattr__("_embedder_" + subnetwork_type, GAT(hparams))
                self._embedder[subnetwork_type] = self.__getattr__("_embedder_" + subnetwork_type)
            else:
                raise Exception(f"hparams.embedder[{subnetwork_type}]] must be one of ['GAT']")

        if hparams.classifier == "Dense":
            hparams.embedding_dim = hparams.embedding_dim * len(hparams.embedder)
            self._classifier = Dense(hparams)
        else:
            raise Exception("hparams.classifier must be one of {'Dense'}")

        self.criterion = ClassificationLoss(n_classes=hparams.n_classes, loss_type=hparams.loss_type)

    def forward(self, X):
        encodings = self._encoder["Protein_seqs"](X["Protein_seqs"])

        embeddings = []
        for subnetwork_type, _ in self.hparams.embedder.items():
            if X[subnetwork_type].dim() > 2:
                X[subnetwork_type] = X[subnetwork_type][0].squeeze(0)
            embeddings.append(self._embedder[subnetwork_type](encodings, X[subnetwork_type]))

        embeddings = torch.cat(embeddings, 1)

        y_pred = self._classifier(embeddings)
        return y_pred

    def loss(self, Y_hat: torch.Tensor, Y, weights=None):
        Y = Y.type_as(Y_hat)
        if isinstance(weights, torch.Tensor):
            idx = torch.nonzero(weights).view(-1)
        else:
            idx = torch.tensor(np.nonzero(weights)[0])

        Y = Y[idx, :]
        Y_hat = Y_hat[idx, :]

        return self.criterion(Y_hat, Y, use_hierar=False, multiclass=True,
                              hierar_penalty=None, hierar_paras=None, hierar_relations=None)
