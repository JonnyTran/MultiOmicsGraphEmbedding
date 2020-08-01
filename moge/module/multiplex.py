import copy

import pandas as pd
import torch
from transformers import AlbertConfig

from moge.module.classifier import DenseClassification, HierarchicalAWX
from moge.module.embedder import GAT, GCN, GraphSAGE, MultiplexLayerAttention, MultiplexNodeAttention, \
    ExpandedMultiplexGAT
from moge.module.enc_emb_cls import EncoderEmbedderClassifier, remove_self_loops
from moge.module.encoder import ConvLSTM, AlbertEncoder, NodeIDEmbedding
from moge.module.losses import ClassificationLoss, get_hierar_relations
from moge.module.utils import filter_samples


class MultiplexEmbedder(EncoderEmbedderClassifier):
    def __init__(self, hparams):
        torch.nn.Module.__init__(self)

        assert isinstance(hparams.encoder,
                          dict), "hparams.encoder must be a dict. If not multi node types, use MonoplexEmbedder instead."
        assert isinstance(hparams.embedder,
                          dict), "hparams.embedder must be a dict. If not multi-layer, use MonoplexEmbedder instead."
        self.hparams = hparams

        ################### Encoding ####################
        self.node_types = list(hparams.encoder.keys())
        for node_type, encoder in hparams.encoder.items():
            if encoder == "ConvLSTM":
                assert not (len(hparams.encoder) > 1 and not len(hparams.vocab_size) > 1)
                self.set_encoder(node_type, ConvLSTM(hparams))

            elif encoder == "Albert":
                assert not (len(hparams.encoder) > 1 and not len(hparams.vocab_size) > 1)
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
                self.set_encoder(node_type, AlbertEncoder(config))

            elif "NodeIDEmbedding" in encoder:
                # `encoder` is a dict with {"NodeIDEmbedding": hparams}
                self.set_encoder(node_type, NodeIDEmbedding(hparams=encoder["NodeIDEmbedding"]))
            elif "Linear" in encoder:
                encoder_hparams = encoder["Linear"]
                self.set_encoder(node_type, torch.nn.Linear(in_features=encoder_hparams["in_features"],
                                                            out_features=hparams.encoding_dim))

            else:
                raise Exception("hparams.encoder must be one of {'ConvLSTM', 'Albert', 'NodeIDEmbedding'}")

        ################### Layer-specfic Embedding ####################
        for subnetwork_type, embedder_model in hparams.embedder.items():
            if embedder_model == "GAT":
                self.set_embedder(subnetwork_type, GAT(hparams))
            elif embedder_model == "GCN":
                self.set_embedder(subnetwork_type, GCN(hparams))
            elif embedder_model == "GraphSAGE":
                self.set_embedder(subnetwork_type, GraphSAGE(hparams))
            else:
                raise Exception(
                    f"Embedder model for hparams.embedder[{subnetwork_type}]] must be one of ['GAT', 'GCN', 'GraphSAGE']")

        ################### Multiplex Embedding ####################
        layers = list(hparams.embedder.keys())
        self.layers = layers
        if hparams.multiplex_embedder == "MultiplexLayerAttention":
            self._multiplex_embedder = MultiplexLayerAttention(embedding_dim=hparams.embedding_dim,
                                                               hidden_dim=hparams.multiplex_hidden_dim,
                                                               attention_dropout=hparams.multiplex_attn_dropout,
                                                               layers=layers)
            hparams.embedding_dim = hparams.multiplex_hidden_dim
        elif hparams.multiplex_embedder == "MultiplexNodeAttention":
            self._multiplex_embedder = MultiplexNodeAttention(embedding_dim=hparams.embedding_dim,
                                                              hidden_dim=hparams.multiplex_hidden_dim,
                                                              attention_dropout=hparams.multiplex_attn_dropout,
                                                              layers=layers)
            hparams.embedding_dim = hparams.multiplex_hidden_dim
        else:
            print('"multiplex_embedder" not used. Concatenate multi-layer embeddings instead.')
            hparams.embedding_dim = hparams.embedding_dim * len(hparams.embedder)

        ################### Classifier ####################
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

        self.criterion = ClassificationLoss(
            n_classes=hparams.n_classes,
            class_weight=None if not hasattr(hparams, "class_weight") else torch.tensor(hparams.class_weight),
            loss_type=hparams.loss_type,
            hierar_penalty=hparams.hierar_penalty if hparams.use_hierar else None,
            hierar_relations=hierar_relations if hparams.use_hierar else None
        )

    def forward(self, X):
        if X[self.node_types[0]].dim() > 2:
            X[self.node_types[0]] = X[self.node_types[0]].squeeze(0)

        encodings = self.get_encoder(self.node_types[0]).forward(X[self.node_types[0]])

        embeddings = []
        for layer in self.layers:
            if X[layer].dim() > 2:
                X[layer] = X[layer].squeeze(0)
                X[layer], _ = remove_self_loops(X[layer], None)
            embeddings.append(self.get_embedder(layer).forward(encodings, X[layer]))

        if hasattr(self, "_multiplex_embedder"):
            embeddings = self._multiplex_embedder.forward(embeddings)
        else:
            embeddings = torch.cat(embeddings, dim=1)

        y_pred = self._classifier(embeddings)
        return y_pred

    def loss(self, Y_hat: torch.Tensor, Y, weights=None):
        Y_hat, Y = filter_samples(Y_hat, Y, weights)

        return self.criterion.forward(
            Y_hat, Y,
            use_hierar=self.hparams.use_hierar,
            multiclass=False if "SOFTMAX" in self.hparams.loss_type else True,
            classifier_weight=self._classifier.fc_classifier.linear.att_weight if self.hparams.use_hierar else None,
        )

    def get_embeddings(self, X, batch_size=100, return_multi_emb=False):
        """
        Get embeddings for a set of nodes in `X`.
        :param X: a dict with keys {"input_seqs", "subnetwork"}
        :param cuda (bool): whether to run computations in GPUs
        :return (np.array): a numpy array of size (node size, embedding dim)
        """
        if X["Protein_seqs"].dim() > 2:
            X["Protein_seqs"] = X["Protein_seqs"].squeeze(0)

        encodings = self.get_encodings(X, node_type="Protein_seqs", batch_size=batch_size)

        multi_embeddings = []
        for layer, _ in self.hparams.embedder.items():
            if X[layer].dim() > 2:
                X[layer] = X[layer].squeeze(0)
                X[layer], _ = remove_self_loops(X[layer], None)
            multi_embeddings.append(self.get_embedder(layer).forward(encodings, X[layer]))

        if return_multi_emb:
            return multi_embeddings

        if hasattr(self, "_multiplex_embedder"):
            embeddings = self._multiplex_embedder.forward(multi_embeddings)
        else:
            embeddings = torch.cat(multi_embeddings, 1)

        return embeddings.detach().cpu().numpy()


class HeterogeneousMultiplexEmbedder(MultiplexEmbedder):
    def __init__(self, hparams):
        torch.nn.Module.__init__(self)

        assert isinstance(hparams.encoder,
                          dict), "hparams.encoder must be a dict. If not multi node types, use MonoplexEmbedder instead."
        assert isinstance(hparams.embedder,
                          dict), "hparams.embedder must be a dict. If not multi-layer, use MonoplexEmbedder instead."
        self.hparams = copy.copy(hparams)

        ################### Encoding ####################
        self.node_types = list(hparams.encoder.keys())
        for node_type, encoder in hparams.encoder.items():
            if encoder == "ConvLSTM":
                hparams.vocab_size = self.hparams.vocab_size[node_type]
                self.set_encoder(node_type, ConvLSTM(hparams))

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
                self.set_encoder(node_type, AlbertEncoder(config))

            elif "NodeIDEmbedding" in encoder:
                # `encoder` is a dict with {"NodeIDEmbedding": hparams}
                self.set_encoder(node_type, NodeIDEmbedding(hparams=encoder["NodeIDEmbedding"]))

            elif "Linear" in encoder:
                encoder_hparams = encoder["Linear"]
                self.set_encoder(node_type, torch.nn.Linear(in_features=encoder_hparams["in_features"],
                                                            out_features=hparams.encoding_dim))

            else:
                raise Exception("hparams.encoder must be one of {'ConvLSTM', 'Albert', 'NodeIDEmbedding'}")

        ################### Layer-specfic Embedding ####################
        self.layers = list(hparams.embedder)
        if hparams.multiplex_embedder == "ExpandedMultiplexGAT":
            self._embedder = ExpandedMultiplexGAT(in_channels=hparams.encoding_dim,
                                                  out_channels=int(
                                                      hparams.embedding_dim / len(self.node_types)),
                                                  node_types=self.node_types,
                                                  layers=self.layers,
                                                  dropout=hparams.nb_attn_dropout)
        else:
            print('"multiplex_embedder" used. Concatenate multi-layer embeddings instead.')

        ################### Classifier ####################
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

        self.criterion = ClassificationLoss(
            n_classes=hparams.n_classes,
            class_weight=None if not hasattr(hparams, "class_weight") else torch.tensor(hparams.class_weight),
            loss_type=hparams.loss_type,
            hierar_penalty=hparams.hierar_penalty if hparams.use_hierar else None,
            hierar_relations=hierar_relations if hparams.use_hierar else None
        )

    def forward(self, X):
        X = {key: X[key].squeeze(0) if X[key].dim() > 2 else X[key] for key in X}
        batch_size = X[self.node_types[0]].size(0)

        encodings = {}
        for node_type in self.node_types:
            # nonzero_index = X[node_type].sum(1) > 0
            # print("nonzero_index", nonzero_index)
            # inputs = X[node_type][nonzero_index, :]
            encodings[node_type] = self.get_encoder(node_type).forward(X[node_type])
            # print(f"{node_type}, {encodings[node_type].shape}")

        sample_idx_by_type = {}
        index = 0
        for i, node_type in enumerate(self.node_types):
            sample_idx_by_type[node_type] = index
            index += encodings[node_type].size(0)

        embeddings = self._embedder.forward(x=encodings,
                                            sample_idx_by_type=sample_idx_by_type,
                                            edge_index={layer: X[layer] for layer in self.layers})
        # print("embeddings", embeddings.shape)

        merge_embeddings = []
        for i, node_type in enumerate(self.node_types):
            merge_embeddings.append(
                embeddings[sample_idx_by_type[node_type]: sample_idx_by_type[node_type] + batch_size])

        merge_embeddings = torch.cat(merge_embeddings, dim=1)
        # print("merge_embeddings", merge_embeddings.shape)

        y_pred = self._classifier(merge_embeddings)
        return y_pred

    def get_embeddings(self, X, batch_size=100, return_multi_emb=False):
        """
        Get embeddings for a set of nodes in `X`.
        :param X: a dict with keys {"input_seqs", "subnetwork"}
        :param cuda (bool): whether to run computations in GPUs
        :return (np.array): a numpy array of size (node size, embedding dim)
        """
        X = {key: X[key].squeeze(0) if X[key].dim() > 2 else X[key] for key in X}
        batch_size = X[self.node_types[0]].size(0)

        encodings = {}
        for node_type in self.node_types:
            encodings[node_type] = self.get_encodings(X, node_type, batch_size)

        sample_idx_by_type = {}
        index = 0
        for i, node_type in enumerate(self.node_types):
            sample_idx_by_type[node_type] = index
            index += encodings[node_type].size(0)

        embeddings = self._embedder.forward(x=encodings,
                                            sample_idx_by_type=sample_idx_by_type,
                                            edge_index={layer: X[layer] for layer in self.layers})
        # print("embeddings", embeddings.shape)

        merge_embeddings = []
        for i, node_type in enumerate(self.node_types):
            merge_embeddings.append(
                embeddings[sample_idx_by_type[node_type]: sample_idx_by_type[node_type] + batch_size])

        merge_embeddings = torch.cat(merge_embeddings, dim=1)

        return merge_embeddings.detach().cpu().numpy()
