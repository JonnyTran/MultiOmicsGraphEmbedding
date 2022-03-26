import pandas as pd
import torch
from moge.model.networkx.embedder import GAT
from moge.model.networkx.encoder import ConvLSTM, AlbertEncoder
from transformers import AlbertConfig

from moge.model.classifier import DenseClassification, HierarchicalAWX
from moge.model.losses import ClassificationLoss, get_hierar_relations
from moge.model.utils import filter_samples


class EncoderEmbedderClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super(EncoderEmbedderClassifier).__init__()

    def get_encoder(self, node_type):
        return self.__getattr__("_encoder_" + node_type)

    def set_encoder(self, node_type, model):
        self.__setattr__("_encoder_" + node_type, model)

    def get_embedder(self, layer):
        return self.__getattr__("_embedder_" + layer)

    def set_embedder(self, layer, model):
        self.__setattr__("_embedder_" + layer, model)

    def get_embeddings(self, *args):
        raise NotImplementedError()

    def get_encodings(self, X, node_type, batch_size=32):
        if node_type is not None:
            input_seqs = X[node_type]

        if isinstance(self.hparams.encoder, dict):
            encoder_module = self.get_encoder(node_type)
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

    def predict(self, embeddings):
        y_pred = self._classifier.forward(embeddings, None)
        if "LOGITS" in self.hparams.loss_type:
            if "SOFTMAX" in self.hparams.loss_type:
                print("INFO: Applied softmax to predictions")
                y_pred = torch.softmax(y_pred, 1)
            else:
                print("INFO: Applied sigmoid to predictions")
                y_pred = torch.sigmoid(y_pred)

        return y_pred.detach().cpu().numpy()


class MonoplexEmbedder(EncoderEmbedderClassifier):
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
            self._classifier = DenseClassification(hparams)
        elif hparams.classifier == "HierarchicalAWX":
            self._classifier = HierarchicalAWX(hparams)
        else:
            raise Exception("hparams.classifier must be one of {'Dense'}")

        if hparams.use_hierar:
            label_map = pd.Series(range(len(hparams.classes)), index=hparams.classes).to_dict()
            hierar_relations = get_hierar_relations(hparams.hierar_taxonomy_file,
                                                    label_map=label_map)

        self.criterion = ClassificationLoss(n_classes=hparams.n_classes, loss_type=hparams.loss_type,
                                            hierar_penalty=hparams.hierar_penalty if hparams.use_hierar else None,
                                            hierar_relations=hierar_relations if hparams.use_hierar else None)
        self.hparams = hparams

    def forward(self, X):
        input_seqs, subnetwork = X["input_seqs"], X["subnetwork"]
        if subnetwork.dim() > 2:
            subnetwork = subnetwork.squeeze(0)
            subnetwork, _ = remove_self_loops(subnetwork, None)
        encodings = self._encoder(input_seqs)
        embeddings = self._embedder(encodings, subnetwork)
        y_pred = self._classifier(embeddings)
        return y_pred

    def loss(self, Y_hat: torch.Tensor, Y, weights=None):
        Y_hat, Y = filter_samples(Y_hat, Y, weights)

        return self.criterion.forward(Y_hat, Y,
                                      dense_weight=self._classifier.fc_classifier.linear_l.att_weight if self.hparams.use_hierar else None)

    def get_embeddings(self, X, batch_size=None):
        """
        Get embeddings for a set of nodes in `X`.
        :param X: a dict with keys {"input_seqs", "subnetwork"}
        :param cuda (bool): whether to run computations in GPUs
        :return (np.array): a numpy array of size (node size, embedding dim)
        """
        encodings = self.get_encodings(X, node_type="input_seqs", batch_size=batch_size)

        embeddings = self._embedder(encodings, X["subnetwork"])

        return embeddings.detach().cpu().numpy()


def remove_self_loops(edge_index, edge_attr=None):
    r"""Removes every self-loop in the graph given by :attr:`edge_index`, so
    that :math:`(i,i) \not\in \mathcal{E}` for every :math:`i \in \mathcal{V}`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    row, col = edge_index
    mask = row != col
    edge_attr = edge_attr if edge_attr is None else edge_attr[mask]
    edge_index = edge_index[:, mask]

    return edge_index, edge_attr
