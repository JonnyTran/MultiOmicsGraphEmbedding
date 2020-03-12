import numpy as np
import pandas as pd
import tensorflow as tf

from moge.network.multiplex import MultiplexAttributedNetwork
from .subgraph_generator import SubgraphGenerator


class MultiplexGenerator(SubgraphGenerator):
    def __init__(self, network: MultiplexAttributedNetwork, node_list, variables: list = None, targets: list = None,
                 batch_size=500,
                 sampling='neighborhood', compression_func="log", n_steps=100, directed=None,
                 maxlen=1400, padding='post', truncating='post', sequence_to_matrix=False, tokenizer=None,
                 replace=True, seed=0, verbose=True, **kwargs):

        super(MultiplexGenerator, self).__init__(network=network, variables=variables, targets=targets,
                                                 batch_size=batch_size,
                                                 sampling=sampling, compression_func=compression_func, n_steps=n_steps,
                                                 directed=directed, maxlen=maxlen,
                                                 padding=padding, truncating=truncating,
                                                 sequence_to_matrix=sequence_to_matrix, tokenizer=tokenizer,
                                                 replace=replace, seed=seed, verbose=verbose, node_list=node_list,
                                                 **kwargs)

    def get_output_types(self):
        return ({"input_seqs": tf.int8, "subnetwork": tf.float32},) + (tf.float32,) * len(self.variables) + \
               (tf.int64,  # y
                tf.bool)  # idx_weights

    def get_output_shapes(self):
        return ({"input_seqs": tf.TensorShape([None, None]),
                 "subnetwork": tf.TensorShape([None, None])},) + \
               (tf.TensorShape([None, None]),) * len(self.variables) + \
               (tf.TensorShape([None, None]),  # y
                tf.TensorShape((None)))  # idx_weights

    def process_sampling_table(self, network):
        self.edge_dict = {}
        self.edge_counts_dict = {}

        self.node_degrees = pd.Series(0, index=network.node_list)
        for modality, network_layer in network.networks.items():
            layer_node_degrees = pd.Series(dict(network_layer.degree(network.node_list)))
            self.node_degrees += layer_node_degrees
        self.node_degrees = self.node_degrees.to_dict()

        self.node_degrees_list = [self.node_degrees[node] if node in self.node_degrees else 0 for node in
                                  self.node_list]
        self.node_sampling_freq = self.compute_node_sampling_freq(self.node_degrees_list,
                                                                  compression=self.compression_func)
        print("# of nodes to sample from (non-zero degree):",
              np.count_nonzero(self.node_sampling_freq)) if self.verbose else None
        assert len(self.node_sampling_freq) == len(self.node_list)

    def __getitem__(self, item=None):
        sampled_nodes = self.sample_node_list(batch_size=self.batch_size)
        X, y, idx_weights = self.__getdata__(sampled_nodes, variable_length=False)

        return X, y, idx_weights

    def __getdata__(self, sampled_nodes, variable_length=False):
        # Features
        X = {}
        X["input_seqs"] = self.get_sequences(sampled_nodes, variable_length=variable_length)
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
