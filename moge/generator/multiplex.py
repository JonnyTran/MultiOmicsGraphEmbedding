from collections import OrderedDict
from itertools import islice

import networkx as nx
import numpy as np
import pandas as pd
import tensorflow as tf

from moge.network.multiplex import MultiplexAttributedNetwork
from .sequences import MultiSequenceTokenizer
from .subgraph_generator import SubgraphGenerator


class MultiplexGenerator(SubgraphGenerator, MultiSequenceTokenizer):
    def __init__(self, network: MultiplexAttributedNetwork, variables: list = [], targets: list = None,
                 batch_size=500, traversal='neighborhood', traversal_depth=2, sampling="log", n_steps=100,
                 maxlen=1400, padding='post', truncating='post', agg_mode=False, tokenizer=None,
                 replace=True, seed=0, verbose=True, **kwargs):

        super(MultiplexGenerator, self).__init__(network=network,
                                                 variables=variables, targets=targets,
                                                 batch_size=batch_size,
                                                 traversal=traversal, traversal_depth=traversal_depth,
                                                 sampling=sampling,
                                                 n_steps=n_steps,
                                                 directed=None, maxlen=maxlen,
                                                 padding=padding, truncating=truncating,
                                                 agg_mode=agg_mode, tokenizer=tokenizer,
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
        return ({"MicroRNA_seqs": tf.TensorShape([self.batch_size, None]),
                 "MessengerRNA_seqs": tf.TensorShape([self.batch_size, None]),
                 "LncRNA_seqs": tf.TensorShape([self.batch_size, None]),
                 "Protein_seqs": tf.TensorShape([self.batch_size, None]),
                 "MicroRNA-MessengerRNA": tf.TensorShape([self.batch_size, self.batch_size]),
                 "MicroRNA-LncRNA": tf.TensorShape([self.batch_size, self.batch_size]),
                 "LncRNA-MessengerRNA": tf.TensorShape([self.batch_size, self.batch_size]),
                 "Protein-Protein": tf.TensorShape([self.batch_size, self.batch_size])},) + \
               (tf.TensorShape([self.batch_size, None]),  # y
                tf.TensorShape((self.batch_size)))  # idx_weights

    def process_normalized_node_degree(self, network):
        self.node_degrees = pd.Series(0, index=self.node_list)
        for modality, network_layer in network.networks.items():
            layer_node_degrees = pd.Series(dict(network_layer.degree(self.node_list)))
            layer_node_degrees = layer_node_degrees / layer_node_degrees.std()

            self.node_degrees[layer_node_degrees.index] = self.node_degrees[
                                                              layer_node_degrees.index] + layer_node_degrees
        self.node_degrees = self.node_degrees.to_dict()

        self.node_degrees_list = [self.node_degrees[node] if node in self.node_degrees else 0 for node in
                                  self.node_list]
        self.node_sampling_freq = self.normalize_node_degrees(self.node_degrees_list,
                                                              compression=self.sampling)
        print("# of nodes to sample from (non-zero degree):",
              np.count_nonzero(self.node_sampling_freq)) if self.verbose else None
        assert len(self.node_sampling_freq) == len(self.node_list)

    def node_sampling(self, batch_size):
        sampled_nodes = self.sample_seed_node(batch_size).tolist()

        while len(sampled_nodes) < batch_size:
            add_nodes = self.sample_seed_node(batch_size - len(sampled_nodes)).tolist()
            sampled_nodes = list(OrderedDict.fromkeys(sampled_nodes + add_nodes))
        return sampled_nodes

    def bfs_traversal(self, batch_size, seed_node=None):
        sampled_nodes = []

        while len(sampled_nodes) < batch_size:
            if seed_node is None or seed_node not in self.node_list:
                start_node = self.sample_seed_node(1)[0]
            else:
                start_node = seed_node

            successor_nodes = []
            for modality, network_layer in self.network.networks.items():
                if start_node not in network_layer.nodes:
                    continue
                layer_neighbors = [node for source, successors in
                                   islice(nx.traversal.bfs_successors(network_layer, source=start_node),
                                          self.traversal_depth) for node in successors]

                if len(layer_neighbors) > batch_size / len(self.network.networks):
                    layer_neighbors = layer_neighbors[:int(batch_size // len(self.network.networks))]
                successor_nodes.extend(layer_neighbors)

            sampled_nodes.extend([start_node] + successor_nodes)
            sampled_nodes = list(OrderedDict.fromkeys(sampled_nodes))

        if len(sampled_nodes) > batch_size:
            np.random.shuffle(sampled_nodes)
            sampled_nodes = sampled_nodes[:batch_size]
        return sampled_nodes

    def dfs_traversal(self, batch_size, seed_node=None):
        sampled_nodes = []

        while len(sampled_nodes) < batch_size:
            if seed_node is None or seed_node not in self.node_list:
                start_node = self.sample_seed_node(1)[0]
            else:
                start_node = seed_node

            successor_nodes = []
            for modality, network_layer in self.network.networks.items():
                if start_node not in network_layer.nodes:
                    continue
                layer_neighbors = list(
                    islice(nx.traversal.dfs_successors(network_layer, source=start_node), batch_size))

                if len(layer_neighbors) > batch_size / len(self.network.networks):
                    layer_neighbors = layer_neighbors[:int(batch_size // len(self.network.networks))]
                successor_nodes.extend(layer_neighbors)

            sampled_nodes.extend([start_node] + successor_nodes)
            sampled_nodes = list(OrderedDict.fromkeys(sampled_nodes))

        if len(sampled_nodes) > batch_size:
            np.random.shuffle(sampled_nodes)
            sampled_nodes = sampled_nodes[:batch_size]
        return sampled_nodes

    def __getitem__(self, item=None):
        sampled_nodes = self.traverse_network(batch_size=self.batch_size)
        X, y, idx_weights = self.__getdata__(sampled_nodes, variable_length=False)

        return X, y, idx_weights

    def __getdata__(self, sampled_nodes, variable_length=False):
        # Features
        X = {}
        for modality in self.network.modalities:
            X["_".join([modality, "seqs"])] = self.get_sequence_encodings(sampled_nodes, modality=modality,
                                                                          variable_length=variable_length or self.variable_length,
                                                                          minlen=40)
            for variable in self.variables:
                if "expression" == variable:
                    X["_".join([modality, variable])] = self.get_expressions(sampled_nodes, modality=modality)
                else:
                    labels_vector = self.annotations[modality].loc[sampled_nodes, variable]
                    labels_vector = self.process_label(labels_vector)
                    X["_".join([modality, variable])] = self.network.feature_transformer[variable].transform(
                        labels_vector)

        for layer, network_layer in self.network.networks.items():
            layer_key = "-".join(layer)
            X[layer_key] = self.network.get_adjacency_matrix(edge_types=layer, node_list=sampled_nodes,
                                                             method=self.method,
                                                             output=self.adj_output)

        # Labels
        targets_vector = self.network.all_annotations.loc[sampled_nodes, self.targets[0]]
        targets_vector = self.process_label(targets_vector)

        try:
            y = self.network.feature_transformer[self.targets[0]].transform(targets_vector)
            if self.sparse_target == 1:
                y = self.label_sparsify(y)[[0]]  # Select only a single label
            elif self.sparse_target == True:
                y = self.label_sparsify(y)  # Select all multilabels

        except Exception as e:
            print("targets_vector", targets_vector.shape, targets_vector.notnull().sum(), targets_vector)
            print("self.network.all_annotations.loc[sampled_nodes, self.targets[0]]",
                  self.network.all_annotations.loc[sampled_nodes, self.targets[0]].shape,
                  self.network.all_annotations.loc[sampled_nodes, self.targets[0]].notnull().sum())
            raise e

        # Get a vector of nonnull indicators
        idx_weights = self.network.all_annotations.loc[sampled_nodes, self.targets].notnull().any(axis=1).values * 1

        assert len(sampled_nodes) == y.shape[0]
        return X, y, idx_weights

    def load_data(self, connected_nodes_only=True, dropna=True, y_label=None, variable_length=False):
        if connected_nodes_only:
            node_list = self.get_connected_nodelist()
        else:
            node_list = self.network.node_list

        if dropna:
            node_list = self.network.all_annotations.loc[node_list, self.targets].dropna().index.tolist()
        else:
            node_list = node_list

        X, y, idx_weights = self.__getdata__(node_list, variable_length=variable_length)

        if y_label:
            y_labels = self.get_node_labels(y_label, node_list=node_list)
            return X, y_labels

        y = pd.DataFrame(y, index=node_list,
                         columns=self.network.feature_transformer[self.targets[0]].classes_)
        return X, y, idx_weights
