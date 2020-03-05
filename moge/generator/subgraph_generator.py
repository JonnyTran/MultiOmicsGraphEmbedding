from collections import OrderedDict

import numpy as np
import pandas as pd

from .sampled_generator import SampledDataGenerator


class SubgraphGenerator(SampledDataGenerator):
    def __init__(self, network, variables: list = None, targets: list = None, batch_size=500,
                 sampling_method='neighborhood_sampling', compression_func="log", n_steps=100, directed=True,
                 maxlen=1400, padding='post', truncating='post', sequence_to_matrix=False, tokenizer=None, replace=True,
                 seed=0, verbose=True, **kwargs):
        """
        Samples a batch subnetwork for classification task.

        :param network: a HeterogeneousNetwork object
        :param variables (list): list of annotation column names as features
        :param targets (list): list of annotation column names to prediction target
        :param batch_size: number of nodes to sample each batch
        :param sampling_method: {'node_sampling', 'neighborhood_sampling', 'all'}. If 'all', overrides batch_size and returns the
        :param compression_func: {"log", "sqrt", "linear"}
        :param n_steps:
        :param directed:
        :param maxlen:
        :param seed:
        :param verbose:
        """
        super(SubgraphGenerator, self).__init__(network=network, variables=variables, targets=targets,
                                                batch_size=batch_size,
                                                sampling_method=sampling_method, compression_func=compression_func,
                                                n_steps=n_steps, directed=directed, replace=replace,
                                                maxlen=maxlen, padding=padding, truncating=truncating,
                                                sequence_to_matrix=sequence_to_matrix,
                                                tokenizer=tokenizer, seed=seed, verbose=verbose, **kwargs)

    def __getitem__(self, item=None):
        sampled_nodes = self.sample_node_list(batch_size=self.batch_size)
        X, y, idx_weights = self.__getdata__(sampled_nodes)

        return X, y, idx_weights

    def sample_node_list(self, batch_size):
        if self.sampling_method == "node_sampling":
            return self.node_sampling(batch_size)
        elif self.sampling_method == "neighborhood_sampling":
            return self.neighborhood_sampling(batch_size)
        elif self.sampling_method == "all":
            return self.network.node_list
        else:
            raise Exception("self.sampling_method must be {'node_sampling', 'neighborhood_sampling', 'all'}")

    def node_sampling(self, batch_size):
        sampled_nodes = self.sample_node(batch_size)
        while len(sampled_nodes) < batch_size:
            add_nodes = np.random.choice(self.node_list, size=batch_size - len(sampled_nodes), replace=False,
                                         p=self.node_sampling_freq).tolist()
            sampled_nodes = list(OrderedDict.fromkeys(sampled_nodes + add_nodes))
        return sampled_nodes

    def neighborhood_sampling(self, batch_size):
        sampled_nodes = []

        while len(sampled_nodes) < batch_size:
            seed_node = self.sample_node(1)
            sampled_nodes = sampled_nodes + list(seed_node) + list(self.network.G.neighbors(seed_node[0]))
            sampled_nodes = list(OrderedDict.fromkeys(sampled_nodes))

        if len(sampled_nodes) > batch_size:
            sampled_nodes = sampled_nodes[:batch_size]
        return sampled_nodes

    def __getdata__(self, sampled_nodes):
        # Features
        X = {}
        X["input_seqs"] = self.get_sequence_data(sampled_nodes, variable_length=False)
        # X["subnetwork"] = self.network.get_graph_laplacian(edge_types=["d"], node_list=sampled_nodes)
        X["subnetwork"] = self.network.get_adjacency_matrix(edge_types=["d"] if self.directed else ["u"],
                                                            node_list=sampled_nodes).toarray()
        X["subnetwork"] = X["subnetwork"] + np.eye(X["subnetwork"].shape[0])  # Add self-loops

        # Features
        for variable in self.variables:
            if "expression" in variable:
                X[variable] = self.get_expressions(sampled_nodes, modality="Protein")

            else:
                labels_vector = self.annotations.loc[sampled_nodes, variable]
                if labels_vector.dtypes == np.object:
                    if labels_vector.str.contains("|").any():
                        labels_vector = labels_vector.str.split("|")
                        labels_vector = labels_vector.map(lambda x: x if isinstance(x, list) else [])
                else:
                    labels_vector = labels_vector.to_numpy().reshape(-1, 1)
                X[variable] = self.network.feature_transformer[variable].transform(labels_vector)

        # Labels
        targets_vector = self.annotations.loc[sampled_nodes, self.targets[0]]
        if targets_vector.dtypes == np.object:
            if targets_vector.str.contains("|").any():
                targets_vector = targets_vector.str.split("|")
                targets_vector = targets_vector.map(lambda x: x if type(x) == list else [])
        else:
            targets_vector = targets_vector.to_numpy().reshape(-1, 1)

        y = self.network.feature_transformer[self.targets[0]].transform(targets_vector)

        # Get a vector of nonnull indicators
        idx_weights = self.annotations.loc[sampled_nodes, self.targets].notnull().any(axis=1)

        # Make a probability distribution
        # y = (1 / y.sum(axis=1)).reshape(-1, 1) * y

        assert len(sampled_nodes) == y.shape[0]
        return X, y, idx_weights

    def load_data(self, connected_nodes_only=True, dropna=True, y_label=None):
        if connected_nodes_only:
            node_list = self.get_nonzero_nodelist()
        else:
            node_list = self.network.node_list

        if dropna:
            node_list = self.annotations.loc[node_list, self.variables + self.targets].dropna().index.tolist()
        else:
            node_list = node_list

        X, y, idx_weights = self.__getdata__(node_list)

        if y_label:
            y_labels = self.get_node_labels(y_label, node_list=node_list)
            return X, y_labels

        y = pd.DataFrame(y, index=node_list,
                         columns=self.network.feature_transformer[self.targets[0]].classes_)
        return X, y, idx_weights
