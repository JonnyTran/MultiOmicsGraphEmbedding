from collections import OrderedDict
from itertools import islice

import networkx as nx
import numpy as np
import pandas as pd
import tensorflow as tf

from .sampled_generator import SampledDataGenerator


class SubgraphGenerator(SampledDataGenerator):
    def __init__(self, network, variables: list = None, targets: list = None, batch_size=500,
                 sampling='neighborhood', compression="log", n_steps=100, directed=True,
                 maxlen=1400, padding='post', truncating='post', seq2array=False, tokenizer=None, replace=True,
                 seed=0, verbose=True, **kwargs):
        """
        Samples a batch subnetwork for classification task.

        :param network: a HeterogeneousNetwork object
        :param variables (list): list of annotation column names as features
        :param targets (list): list of annotation column names to prediction target
        :param batch_size: number of nodes to sample each batch
        :param sampling: {'node', 'neighborhood', 'all'}. If 'all', overrides batch_size and returns the
        :param compression: {"log", "sqrt", "linear"}
        :param n_steps:
        :param directed:
        :param maxlen:
        :param seed:
        :param verbose:
        """
        super(SubgraphGenerator, self).__init__(network=network, variables=variables, targets=targets,
                                                batch_size=batch_size,
                                                sampling=sampling, compression=compression,
                                                n_steps=n_steps, directed=directed, replace=replace,
                                                maxlen=maxlen, padding=padding, truncating=truncating,
                                                seq2array=seq2array,
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
        sampled_nodes = self.sample_node_list(batch_size=self.batch_size)
        X, y, idx_weights = self.__getdata__(sampled_nodes, variable_length=False)

        return X, y, idx_weights

    def sample_node_list(self, batch_size):
        if self.sampling == "node":
            return self.node_sampling(batch_size)
        elif self.sampling == "neighborhood" or self.sampling == "bfs":
            return self.bfs_traversal(batch_size)
        elif self.sampling == "dfs":
            return self.dfs_traversal(batch_size)
        elif self.sampling == "all":
            return self.network.node_list
        elif self.sampling == 'circle':
            return next(self.node_circle_sampling())
        else:
            raise Exception("self.sampling_method must be {'node', 'neighborhood', 'all'}")

    def node_circle_sampling(self):
        yield [node for node in islice(self.nodes_circle, self.batch_size)]

        yield [node for node in islice(self.nodes_circle, self.batch_size)]

    def node_sampling(self, batch_size):
        sampled_nodes = self.sample_node_by_freq(batch_size)

        while len(sampled_nodes) < batch_size:
            add_nodes = np.random.choice(self.node_list, size=batch_size - len(sampled_nodes), replace=False,
                                         p=self.node_sampling_freq).tolist()
            sampled_nodes = list(OrderedDict.fromkeys(sampled_nodes + add_nodes))
        return sampled_nodes

    def bfs_traversal(self, batch_size):
        sampled_nodes = []

        while len(sampled_nodes) < batch_size:
            seed_node = self.sample_node_by_freq(1)
            successor_nodes = [node for source, successors in
                               islice(nx.traversal.bfs_successors(self.network.G if self.directed else self.network.G_u,
                                                                  source=seed_node[0]),
                                      batch_size) for node in successors]

            sampled_nodes.extend(seed_node.tolist() + successor_nodes)
            sampled_nodes = list(OrderedDict.fromkeys(sampled_nodes))

        if len(sampled_nodes) > batch_size:
            sampled_nodes = sampled_nodes[:batch_size]
        return sampled_nodes

    def dfs_traversal(self, batch_size):
        sampled_nodes = []

        while len(sampled_nodes) < batch_size:
            seed_node = self.sample_node_by_freq(1)
            successor_nodes = list(
                islice(nx.traversal.dfs_successors(self.network.G if self.directed else self.network.G_u,
                                                   source=seed_node[0]), batch_size))
            sampled_nodes.extend(seed_node.tolist() + successor_nodes)
            sampled_nodes = list(OrderedDict.fromkeys(sampled_nodes))

        if len(sampled_nodes) > batch_size:
            sampled_nodes = sampled_nodes[:batch_size]
        return sampled_nodes

    def __getdata__(self, sampled_nodes, variable_length=False):
        # Features
        X = {}
        X["input_seqs"] = self.get_sequence_encodings(sampled_nodes, variable_length=variable_length)
        # X["subnetwork"] = self.network.get_graph_laplacian(edge_types=["d"], node_list=sampled_nodes)
        X["subnetwork"] = self.network.get_adjacency_matrix(edge_types=["d"] if self.directed else ["u"],
                                                            node_list=sampled_nodes).toarray()
        X["subnetwork"] = X["subnetwork"] + np.eye(X["subnetwork"].shape[0])  # Add self-loops

        # Features
        for variable in self.variables:
            if "expression" == variable:
                X[variable] = self.get_expressions(sampled_nodes, modality="Protein")

            else:
                labels_vector = self.annotations.loc[sampled_nodes, variable]
                if labels_vector.dtypes == np.object:
                    if labels_vector.str.contains("|", regex=False).any():
                        labels_vector = labels_vector.str.split("|")
                        labels_vector = labels_vector.map(lambda x: x if isinstance(x, list) else [])
                else:
                    labels_vector = labels_vector.to_numpy().reshape(-1, 1)
                X[variable] = self.network.feature_transformer[variable].transform(labels_vector)

        # Labels
        targets_vector = self.annotations.loc[sampled_nodes, self.targets[0]]
        if targets_vector.dtypes == np.object:
            if targets_vector.str.contains("|", regex=False).any():
                targets_vector = targets_vector.str.split("|")
                targets_vector = targets_vector.map(lambda x: x if type(x) == list else [])
            elif all(targets_vector.isnull()):
                targets_vector = targets_vector.to_numpy().reshape(-1, 1)
        else:
            targets_vector = targets_vector.to_numpy().reshape(-1, 1)

        y = self.network.feature_transformer[self.targets[0]].transform(targets_vector)

        # Get a vector of nonnull indicators
        idx_weights = self.annotations.loc[sampled_nodes, self.targets].notnull().any(axis=1)

        # Make sparse labels in y
        # y_df = pd.DataFrame(y, index=sampled_nodes)
        # y = y_df.apply(lambda x: x.values.nonzero()[0], axis=1)

        # Make a probability distribution
        # y = (1 / y.sum(axis=1)).reshape(-1, 1) * y

        assert len(sampled_nodes) == y.shape[0]
        return X, y, idx_weights

    def load_data(self, connected_nodes_only=True, dropna=True, y_label=None, variable_length=False):
        if connected_nodes_only:
            node_list = self.get_nonzero_nodelist()
        else:
            node_list = self.network.node_list

        if dropna:
            node_list = self.annotations.loc[node_list, self.targets].dropna().index.tolist()
        else:
            node_list = node_list

        X, y, idx_weights = self.__getdata__(node_list, variable_length=variable_length)

        if y_label:
            y_labels = self.get_node_labels(y_label, node_list=node_list)
            return X, y_labels

        y = pd.DataFrame(y, index=node_list,
                         columns=self.network.feature_transformer[self.targets[0]].classes_)
        return X, y, idx_weights
