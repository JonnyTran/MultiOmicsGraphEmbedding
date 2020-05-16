import numpy as np
import torch
from transformers import AlbertConfig

from moge.module.classifier import Dense
from moge.module.embedder import GAT
from moge.module.encoder import ConvLSTM, AlbertEncoder
from moge.module.losses import ClassificationLoss


class EncoderEmbedderClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super(EncoderEmbedderClassifier).__init__()

    def get_embeddings(self, *args):
        raise NotImplementedError()

    def get_encodings(self, X, key, batch_size=None):
        if key is not None:
            input_seqs = X[key]

        if isinstance(self._encoder, dict):
            encoder_module = self._encoder[key]
        else:
            encoder_module = self._encoder

        if batch_size is not None:
            input_chunks = input_seqs.split(split_size=batch_size, dim=0)
            encodings = []
            for i in range(len(input_chunks)):
                encodings.append(encoder_module.forward(input_chunks[i]))
            encodings = torch.cat(encodings, 0)
        else:
            encodings = encoder_module.forward(X)

        return encodings

    def predict(self, embeddings, cuda=True):
        y_pred = self._classifier(embeddings)
        if "LOGITS" in self.hparams.loss_type:
            y_pred = torch.softmax(y_pred, 1) if "SOFTMAX" in self.loss_type else torch.sigmoid(y_pred)

        return y_pred.detach().cpu().numpy()


class MonoplexEmebdder(EncoderEmbedderClassifier):
    def __init__(self, hparams):
        torch.nn.Module.__init__(self)

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

        self.criterion = ClassificationLoss(n_classes=hparams.n_classes, loss_type=hparams.loss_type)
        self.hparams = hparams

    def forward(self, X):
        input_seqs, subnetwork = X["input_seqs"], X["subnetwork"]
        # print("input_seqs", input_seqs.shape)
        # print("subnetwork", len(subnetwork), [batch.shape for batch in subnetwork])
        if subnetwork.dim() > 2:
            subnetwork = subnetwork[0].squeeze(0)

        encodings = self._encoder(input_seqs)
        embeddings = self._embedder(encodings, subnetwork)
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

    def get_embeddings(self, X, batch_size=None):
        """
        Get embeddings for a set of nodes in `X`.
        :param X: a dict with keys {"input_seqs", "subnetwork"}
        :param cuda (bool): whether to run computations in GPUs
        :return (np.array): a numpy array of size (node size, embedding dim)
        """
        encodings = self.get_encodings(X, key="input_seqs", batch_size=batch_size)

        embeddings = self._embedder(encodings, X["subnetwork"])

        return embeddings.detach().cpu().numpy()

