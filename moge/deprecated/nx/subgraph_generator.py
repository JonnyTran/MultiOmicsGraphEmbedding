import warnings

warnings.filterwarnings("ignore")

from collections import OrderedDict
from itertools import islice

import networkx as nx
import numpy as np
import pandas as pd
import tensorflow as tf
from torch.utils import data

from .sampled_generator import SampledDataGenerator


class SubgraphGenerator(SampledDataGenerator, data.Dataset):
    def __init__(self, network, variables: list = None, targets: list = None, batch_size=500,
                 traversal='neighborhood', traversal_depth=2, sampling="log", n_steps=100, directed=True,
                 maxlen=1400, padding='post', truncating='post', agg_mode=None, tokenizer=None, replace=True,
                 variable_length=False, seed=0, verbose=True, **kwargs):
        """
        Samples a subnetwork batch along with variables for classification tasks.

        :param network: a HeterogeneousNetwork object
        :param variables (list): list of annotation column names as features
        :param targets (list): list of annotation column names to prediction target
        :param batch_size (int): number of nodes to sample each batch
        :param traversal (str): {'node', 'neighborhood', 'all'}. If 'all', overrides batch_size and returns the whole `node_list`
        :param sampling (str): {"log", "sqrt", "linear"}
        :param n_steps:
        :param directed:
        :param maxlen:
        :param seed:
        :param verbose:
        """
        self.variable_length = variable_length

        super(SubgraphGenerator, self).__init__(network=network,
                                                variables=variables, targets=targets,
                                                batch_size=batch_size,
                                                traversal=traversal, traversal_depth=traversal_depth, sampling=sampling,
                                                n_steps=n_steps, directed=directed, replace=replace,
                                                maxlen=maxlen, padding=padding, truncating=truncating,
                                                agg_mode=agg_mode,
                                                tokenizer=tokenizer, seed=seed, verbose=verbose, **kwargs)

    def get_output_types(self):
        return ({"input_seqs": tf.int8, "subnetwork": tf.float32},) + \
               (tf.float32,) * len(self.variables) + \
               (tf.int64,  # y
                tf.bool)  # idx_weights

    def get_output_shapes(self):
        return ({"input_seqs": tf.TensorShape([self.batch_size, None]),
                 "subnetwork": tf.TensorShape([self.batch_size, self.batch_size])},) + \
               (tf.TensorShape([self.batch_size, None]),) * len(self.variables) + \
               (tf.TensorShape([self.batch_size, None]),  # y
                tf.TensorShape((self.batch_size)))  # idx_weights

    def __getitem__(self, item=None):
        sampled_nodes = self.traverse_network(batch_size=self.batch_size)
        X, y, idx_weights = self.__getdata__(sampled_nodes, variable_length=False)
        return X, y, idx_weights

    def traverse_network(self, batch_size, seed_node=None):
        if self.traversal == "node":
            return self.node_sampling(batch_size)
        elif self.traversal == "bfs":
            return self.bfs_traversal(batch_size, seed_node=seed_node)
        elif self.traversal == "neighborhood":
            return self.neighbors_traversal(seed_node=seed_node)
        elif self.traversal == "dfs":
            return self.dfs_traversal(batch_size, seed_node=seed_node)
        elif self.traversal == 'all_slices':
            return next(self.iter_node_slices())
        elif self.traversal == "all":
            return self.node_list
        else:
            raise Exception("`sampling` method must be {'node', 'bfs', 'dfs', 'all', or 'all_slices'}")

    def iter_node_slices(self):
        yield [node for node in islice(self.nodes_circle, self.batch_size)]

    def node_sampling(self, batch_size):
        sampled_nodes = self.sample_seed_node(batch_size)

        while len(sampled_nodes) < batch_size:
            add_nodes = np.random.choice(self.node_list, size=batch_size - len(sampled_nodes), replace=False,
                                         p=self.node_sampling_freq).tolist()
            sampled_nodes = list(OrderedDict.fromkeys(sampled_nodes + add_nodes))
        return sampled_nodes

    def neighbors_traversal(self, seed_node):
        if seed_node is None or seed_node not in self.node_list:
            sampled_nodes = self.sample_seed_node(1).tolist()
        else:
            sampled_nodes = [seed_node]

        successor_nodes = [node for source, successors in
                           islice(nx.traversal.bfs_successors(self.network.G if self.directed else self.network.G_u,
                                                              source=sampled_nodes[0]), 1) for node in successors]
        sampled_nodes.extend(successor_nodes)
        sampled_nodes = list(OrderedDict.fromkeys(sampled_nodes))  # remove duplicates

        if len(sampled_nodes) > self.batch_size:
            sampled_nodes = sampled_nodes[:self.batch_size]
        return sampled_nodes

    def bfs_traversal(self, batch_size, seed_node: str = None):
        sampled_nodes = []

        while len(sampled_nodes) < batch_size:
            if seed_node is None or seed_node not in self.node_list:
                start_node = self.sample_seed_node(1)[0]
            else:
                start_node = seed_node

            successor_nodes = [node for source, successors in
                               islice(nx.traversal.bfs_successors(self.network.G if self.directed else self.network.G_u,
                                                                  source=start_node), self.traversal_depth) for node in
                               successors]
            sampled_nodes.extend([start_node] + successor_nodes)
            sampled_nodes = list(OrderedDict.fromkeys(sampled_nodes))

        if len(sampled_nodes) > batch_size:
            sampled_nodes = sampled_nodes[:batch_size]
        return sampled_nodes

    def dfs_traversal(self, batch_size, seed_node: str = None):
        sampled_nodes = []

        while len(sampled_nodes) < batch_size:
            if seed_node is None or seed_node not in self.node_list:
                start_node = self.sample_seed_node(1)[0]
            else:
                start_node = seed_node

            successor_nodes = list(
                islice(nx.traversal.dfs_successors(self.network.G if self.directed else self.network.G_u,
                                                   source=start_node), batch_size))
            sampled_nodes.extend([start_node] + successor_nodes)
            sampled_nodes = list(OrderedDict.fromkeys(sampled_nodes))

        if len(sampled_nodes) > batch_size:
            sampled_nodes = sampled_nodes[:batch_size]
        return sampled_nodes

    def __getdata__(self, sampled_nodes, variable_length=False, training=True):
        # Features
        X = {}
        X["input_seqs"] = self.get_sequence_encodings(sampled_nodes,
                                                      variable_length=variable_length or self.variable_length)

        X["subnetwork"] = self.network.get_adjacency_matrix(edge_types=["d"] if self.directed else ["u"],
                                                            node_list=sampled_nodes, method=self.method,
                                                            output=self.adj_output)

        # Additional Features
        for variable in self.variables:
            if "expression" == variable:
                X[variable] = self.get_expressions(sampled_nodes, modality="Protein")
            else:
                labels_vector = self.process_label(self.annotations.loc[sampled_nodes, variable])
                X[variable] = self.network.feature_transformer[variable].transform(labels_vector)

        # Labels
        targets_vector = self.process_label(self.annotations.loc[sampled_nodes, self.targets[0]])

        y = self.network.feature_transformer[self.targets[0]].transform(targets_vector)
        if self.sparse_target is 1 and training:
            y = self.label_sparsify(y)[[0]]  # Select only a single label
        elif self.sparse_target is True and training:
            y = self.label_sparsify(y)  # Select all multilabels

        # Get a vector of nonnull indicators
        idx_weights = self.annotations.loc[sampled_nodes, self.targets].notnull().any(axis=1).values * 1

        assert len(sampled_nodes) == y.shape[0]
        return X, y, idx_weights

    def load_data(self, connected_nodes_only=True, dropna=True, y_label=None, variable_length=False):
        if connected_nodes_only:
            node_list = self.get_connected_nodelist()
        else:
            node_list = self.network.node_list

        if dropna:
            node_list = self.annotations.loc[node_list, self.targets].dropna().index.tolist()
        else:
            node_list = node_list

        X, y, idx_weights = self.__getdata__(node_list, variable_length=variable_length, training=False)

        if y_label:
            y_labels = self.get_node_labels(y_label, node_list=node_list)
            return X, y_labels

        y = pd.DataFrame(y, index=node_list,
                         columns=self.network.feature_transformer[self.targets[0]].classes_)

        return X, y, idx_weights

