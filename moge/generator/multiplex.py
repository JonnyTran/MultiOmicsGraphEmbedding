from collections import OrderedDict

import numpy as np
import pandas as pd
import tensorflow as tf

from moge.network.multiplex import MultiplexAttributedNetwork
from .sequences import MultiSequenceTokenizer
from .subgraph_generator import SubgraphGenerator


class MultiplexGenerator(SubgraphGenerator, MultiSequenceTokenizer):
    def __init__(self, network: MultiplexAttributedNetwork, variables: list = [], targets: list = None,
                 batch_size=500, sampling='neighborhood', compression_func="log", n_steps=100, directed=None,
                 maxlen=1400, padding='post', truncating='post', seq2array=False, tokenizer=None,
                 replace=True, seed=0, verbose=True, **kwargs):

        super(MultiplexGenerator, self).__init__(network=network, variables=variables, targets=targets,
                                                 batch_size=batch_size,
                                                 sampling=sampling, compression_func=compression_func, n_steps=n_steps,
                                                 directed=directed, maxlen=maxlen,
                                                 padding=padding, truncating=truncating,
                                                 seq2array=seq2array, tokenizer=tokenizer,
                                                 replace=replace, seed=seed, verbose=verbose,
                                                 **kwargs)

    def get_output_types(self):
        return (
               {"MicroRNA_seqs": tf.int8, "MessengerRNA_seqs": tf.int8, "LncRNA_seqs": tf.int8, "Protein_seqs": tf.int8,
                "MicroRNA-MessengerRNA": tf.float32, "MicroRNA-LncRNA": tf.float32, "LncRNA-MessengerRNA": tf.float32,
                "Protein-Protein": tf.float32},) + \
               (tf.int64,  # y
                tf.bool)  # idx_weights

    def get_output_shapes(self):
        return ({"MicroRNA_seqs": tf.TensorShape([None, None]), "MessengerRNA_seqs": tf.TensorShape([None, None]),
                 "LncRNA_seqs": tf.TensorShape([None, None]), "Protein_seqs": tf.TensorShape([None, None]),
                 "MicroRNA-MessengerRNA": tf.TensorShape([None, None]), "MicroRNA-LncRNA": tf.TensorShape([None, None]),
                 "LncRNA-MessengerRNA": tf.TensorShape([None, None]),
                 "Protein-Protein": tf.TensorShape([None, None])},) + \
               (tf.TensorShape([None, None]),  # y
                tf.TensorShape((None)))  # idx_weights

    def process_sampling_table(self, network):
        self.node_degrees = pd.Series(0, index=self.node_list)
        for modality, network_layer in network.networks.items():
            layer_node_degrees = pd.Series(dict(network_layer.degree(self.node_list)))
            self.node_degrees[layer_node_degrees.index] = self.node_degrees[
                                                              layer_node_degrees.index] + layer_node_degrees
        self.node_degrees = self.node_degrees.to_dict()

        self.node_degrees_list = [self.node_degrees[node] if node in self.node_degrees else 0 for node in
                                  self.node_list]
        self.node_sampling_freq = self.compute_node_sampling_freq(self.node_degrees_list,
                                                                  compression=self.compression_func)
        print("# of nodes to sample from (non-zero degree):",
              np.count_nonzero(self.node_sampling_freq)) if self.verbose else None
        assert len(self.node_sampling_freq) == len(self.node_list)

    def neighborhood_sampling(self, batch_size):
        sampled_nodes = []

        while len(sampled_nodes) < batch_size:
            seed_node = self.sample_node_by_freq(1)
            neighbors = []
            for modality, network_layer in self.network.networks.items():
                if seed_node[0] not in network_layer.nodes:
                    continue
                neighbors.extend(list(network_layer.neighbors(seed_node[0])))

            sampled_nodes = sampled_nodes + list(seed_node) + neighbors
            sampled_nodes = [node for node in sampled_nodes if node in self.node_list]
            sampled_nodes = list(OrderedDict.fromkeys(sampled_nodes))

        if len(sampled_nodes) > batch_size:
            sampled_nodes = sampled_nodes[:batch_size]
        return sampled_nodes

    def __getitem__(self, item=None):
        sampled_nodes = self.sample_node_list(batch_size=self.batch_size)
        X, y, idx_weights = self.__getdata__(sampled_nodes, variable_length=False)

        return X, y, idx_weights

    def __getdata__(self, sampled_nodes, variable_length=False):
        # Features
        X = {}
        for modality in self.network.modalities:
            X["_".join([modality, "seqs"])] = self.get_sequence_encodings(sampled_nodes, modality=modality,
                                                                          variable_length=variable_length)
            for variable in self.variables:
                if "expression" == variable:
                    X["_".join([modality, variable])] = self.get_expressions(sampled_nodes, modality=modality)
                else:
                    labels_vector = self.annotations[modality].loc[sampled_nodes, variable]
                    labels_vector = self.process_vector(labels_vector)
                    X["_".join([modality, variable])] = self.network.feature_transformer[variable].transform(
                        labels_vector)

        for layer, network_layer in self.network.networks.items():
            layer_key = "-".join(layer)
            X[layer_key] = self.network.get_adjacency_matrix(edge_types=layer, node_list=sampled_nodes).toarray()
            X[layer_key] = X[layer_key] + np.eye(X[layer_key].shape[0])  # Add self-loops

        # Labels
        targets_vector = self.network.all_annotations.loc[sampled_nodes, self.targets[0]]
        targets_vector = self.process_vector(targets_vector)

        try:
            y = self.network.feature_transformer[self.targets[0]].transform(targets_vector)
        except Exception as e:
            print("targets_vector", targets_vector.shape, targets_vector.notnull().sum(), targets_vector)
            print("self.network.all_annotations.loc[sampled_nodes, self.targets[0]]",
                  self.network.all_annotations.loc[sampled_nodes, self.targets[0]].shape,
                  self.network.all_annotations.loc[sampled_nodes, self.targets[0]].notnull().sum())
            raise e

        # Get a vector of nonnull indicators
        idx_weights = self.network.all_annotations.loc[sampled_nodes, self.targets].notnull().any(axis=1)

        # Make sparse labels in y
        # y_df = pd.DataFrame(y, index=sampled_nodes)
        # y = y_df.apply(lambda x: x.values.nonzero()[0], axis=1)

        # Make a probability distribution
        # y = (1 / y.sum(axis=1)).reshape(-1, 1) * y
        # print("len(sampled_nodes)", len(sampled_nodes))
        # print("y", y.shape)
        # print("idx_weights", idx_weights.shape)
        assert len(sampled_nodes) == y.shape[0]
        return X, y, idx_weights
