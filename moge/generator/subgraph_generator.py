from collections import OrderedDict

import numpy as np
import pandas as pd

from .sampled_generator import SampledDataGenerator


class SubgraphGenerator(SampledDataGenerator):
    def __init__(self, network, variables=None, targets=None, batch_size=500,
                 compression_func="log", n_steps=100, directed=True,
                 maxlen=1400, padding='post', truncating='post', sequence_to_matrix=False, tokenizer=None, replace=True,
                 seed=0, verbose=True, **kwargs):
        """
        Samples a batch subnetwork for classification task.

        :param network: a HeterogeneousNetwork object
        :param variables: list of annotation column names as features
        :param targets: list of annotation column names to prediction target
        :param batch_size: number of nodes to sample each batch
        :param compression_func: {"log", "sqrt", "linear"}
        :param n_steps:
        :param directed:
        :param maxlen:
        :param seed:
        :param verbose:
        """
        self.variables = variables
        self.targets = targets

        super(SubgraphGenerator, self).__init__(network=network, batch_size=batch_size,
                                                compression_func=compression_func, n_steps=n_steps,
                                                directed=directed, replace=replace,
                                                maxlen=maxlen, padding=padding, truncating=truncating,
                                                sequence_to_matrix=sequence_to_matrix,
                                                tokenizer=tokenizer, seed=seed, verbose=verbose, **kwargs)

    def __getitem__(self, item=None):
        sampled_nodes = self.sample_neighborhoods(batch_size=self.batch_size)
        X, y = self.__getdata__(sampled_nodes)

        return X, y

    def sample_subgraph(self, batch_size):
        sampled_nodes = np.random.choice(self.node_list, size=batch_size, replace=False,
                                         p=self.node_sampling_freq)
        sampled_nodes = self.annotations.loc[sampled_nodes, self.variables + self.targets].dropna().index.tolist()
        while len(sampled_nodes) < batch_size:
            add_nodes = np.random.choice(self.node_list, size=batch_size - len(sampled_nodes), replace=False,
                                         p=self.node_sampling_freq).tolist()
            sampled_nodes = list(OrderedDict.fromkeys(sampled_nodes + add_nodes))
            sampled_nodes = self.annotations.loc[
                sampled_nodes, self.variables + self.targets + ["Transcript sequence"]].dropna().index.tolist()
        return sampled_nodes

    def sample_neighborhoods(self, batch_size):
        sampled_nodes = []

        while len(sampled_nodes) < batch_size:
            seed_node = np.random.choice(self.node_list, size=batch_size - len(sampled_nodes), replace=False,
                                         p=self.node_sampling_freq)
            sampled_nodes = sampled_nodes + list(seed_node) + list(self.network.G.neighbors(seed_node[0]))

            sampled_nodes = list(OrderedDict.fromkeys(sampled_nodes))
            sampled_nodes = self.annotations.loc[
                sampled_nodes, self.variables + self.targets + ["Transcript sequence"]].dropna().index.tolist()

        if len(sampled_nodes) > batch_size:
            sampled_nodes = sampled_nodes[:batch_size]
        return sampled_nodes

    def __getdata__(self, sampled_nodes):
        # Features
        X = {}
        X["input_seqs"] = self.get_sequence_data(sampled_nodes, variable_length=False)
        # X["subnetwork"] = self.network.get_graph_laplacian(edge_types=["d"], node_list=sampled_nodes)
        X["subnetwork"] = self.network.get_adjacency_matrix(edge_types=["d"], node_list=sampled_nodes).toarray()
        X["subnetwork"] = X["subnetwork"] + np.eye(X["subnetwork"].shape[0])  # Add self-loops

        for variable in self.variables:
            labels_vector = self.annotations.loc[sampled_nodes, variable]
            if labels_vector.dtypes == np.object:
                if labels_vector.str.contains("|").any():
                    labels_vector = labels_vector.str.split("|")
            else:
                labels_vector = labels_vector.to_numpy().reshape(-1, 1)
            X[variable] = self.network.feature_transformer[variable].transform(labels_vector)

        # Labels
        targets_vector = self.annotations.loc[sampled_nodes, self.targets[0]]
        if targets_vector.dtypes == np.object:
            if targets_vector.str.contains("|").any():
                targets_vector = targets_vector.str.split("|")
        else:
            targets_vector = targets_vector.to_numpy().reshape(-1, 1)
        y = self.network.feature_transformer[self.targets[0]].transform(targets_vector)

        # Make a probability distribution
        # y = (1 / y.sum(axis=1)).reshape(-1, 1) * y

        assert len(sampled_nodes) == y.shape[0]
        return X, y

    def load_data(self, y_label=None):
        sampled_nodes = self.annotations.loc[
            self.get_nonzero_nodelist(), self.variables + self.targets].dropna().index.tolist()
        X, y = self.__getdata__(sampled_nodes)

        if y_label:
            y_labels = self.get_node_labels(y_label, node_list=sampled_nodes)
            return X, y_labels

        y = pd.DataFrame(y, index=sampled_nodes,
                         columns=self.network.feature_transformer[self.targets[0]].classes_)
        return X, y
