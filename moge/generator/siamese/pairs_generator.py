import random

import networkx as nx
import numpy as np
from scipy.sparse import triu

from moge.generator.data_generator import DataGenerator
from moge.generator.utils import EdgelistSampler

IS_DIRECTED = 1
IS_UNDIRECTED = 0
DIRECTED_EDGE = 'd'
UNDIRECTED_EDGE = 'u'
DIRECTED_NEG_EDGE = 'd_n'
UNDIRECTED_NEG_EDGE = 'u_n'


class PairsGenerator(DataGenerator):

    def __init__(self, network, variables=None, targets=None, weighted=False, batch_size=1, replace=True, seed=0,
                 verbose=True, **kwargs):
        self.negative_sampling_ratio = neg_sampling_ratio

        super(PairsGenerator, self).__init__(network=network, weighted=weighted, batch_size=batch_size, replace=replace,
                                             seed=seed, verbose=verbose, maxlen=maxlen, padding=padding,
                                             truncating=truncating, sequence_to_matrix=sequence_to_matrix,
                                             tokenizer=tokenizer)
        self.process_training_edges_data()
        self.process_negative_sampling_edges()

    def process_training_edges_data(self):
        # Directed Edges (regulatory interaction)
        self.adj_directed = self.network.get_adjacency_matrix(edge_types=["d"], node_list=self.node_list)
        self.Ed_rows, self.Ed_cols = self.adj_directed.nonzero()  # getting the list of non-zero edges from the Sparse Numpy matrix
        self.Ed_count = len(self.Ed_rows)

        # Undirected Edges (node similarity)
        self.adj_undirected = self.network.get_adjacency_matrix(edge_types=["u"], node_list=self.node_list)
        self.Eu_rows, self.Eu_cols = triu(self.adj_undirected, k=1).nonzero()
        self.Eu_count = len(self.Eu_rows)

        # Negative Undirected Edges (true negative edges from node similarity)
        self.adj_negative = self.network.get_adjacency_matrix(edge_types=["u_n"], node_list=self.node_list)
        self.En_rows, self.En_cols = triu(self.adj_negative, k=1).nonzero()
        self.En_count = len(self.En_rows)

        print("Ed_count:", self.Ed_count, ", Eu_count:", self.Eu_count, ", En_count:", self.En_count) if self.verbose else None

    def process_negative_sampling_edges(self):
        # All Negative Directed Edges (non-positive edges)
        if self.training_network is not None:
            adj_positive = self.adj_directed + self.adj_undirected + self.training_network.get_adjacency_matrix(
                edge_types=["d"],
                node_list=self.node_list)
        else:
            adj_positive = self.adj_directed + self.adj_undirected
        self.Ens_rows_all, self.Ens_cols_all = np.where(adj_positive.todense() == 0)
        self.Ens_count = int(self.Ed_count * self.negative_sampling_ratio)
        print("Ens_count:", self.Ens_count) if self.verbose else None

    def reload_directed_edges_data(self, edge_types=["d"], databases=None, node_list=None, node_list_B=None):
        if "d" in edge_types:
            self.adj_directed = self.network.get_adjacency_matrix(edge_types=edge_types, node_list=self.node_list,
                                                                  databases=databases)
            self.Ed_rows, self.Ed_cols = self.adj_directed.nonzero()  # getting the list of non-zero edges from the Sparse Numpy matrix
            self.Ed_count = len(self.Ed_rows)

            if node_list is not None and node_list_B is not None:
                self.process_negative_sampling_edges_filtered(node_list, node_list_B)

            self.Ens_count = int(self.Ed_count * self.negative_sampling_ratio)
            print("Ed_count:", self.Ed_count, ", Eu_count:", self.Eu_count, ", En_count:",
                  self.En_count, ", Ens_count", self.Ens_count) if self.verbose else None

        return self.Ed_count + self.Eu_count + self.En_count + self.Ens_count

    def _split_index(self, index):
        'Choose the corresponding edge type data depending on the index number'

        # Index belonging to undirected edges
        if index < self.Ed_count:
            return index, DIRECTED_EDGE

        # Index belonging to undirected edges
        elif self.Ed_count <= index and index < (self.Ed_count + self.Eu_count):
            return index - self.Ed_count, UNDIRECTED_EDGE

        # index belonging to negative edges
        elif (self.Ed_count + self.Eu_count) <= index and index < (self.Ed_count + self.Eu_count + self.En_count):
            return index - (self.Ed_count + self.Eu_count), UNDIRECTED_NEG_EDGE

        # Index belonging to directed negative sampled edges
        elif (self.Ed_count + self.Eu_count + self.En_count) <= index:
            return index - (self.Ed_count + self.Eu_count + self.En_count), DIRECTED_NEG_EDGE
        else:
            raise Exception("Index out of range. Value:" + index)

    def on_epoch_end(self):
        self.update_negative_samples()
        self.annotations["Transcript sequence"] = self.sample_sequences(self.transcripts_to_sample)

        self.indexes = np.arange(self.Ed_count + self.Eu_count + self.En_count + self.Ens_count)

        if self.replace == True:
            np.random.shuffle(self.indexes)

    def get_training_edges(self, training_index):
        """
        Generate training edges (for right now only works with directed edges)
        :param training_index:
        :return:
        """
        # Generate indexes of the batch
        indices = self.indexes[training_index * self.batch_size: (training_index + 1) * self.batch_size]

        # Find list of IDs
        edges_batch = [self._split_index(i) for i in indices]

        X_list = []
        y_list = []
        for id, edge_type in edges_batch:
            if edge_type == DIRECTED_EDGE:
                X_list.append((self.node_list[self.Ed_rows[id]], self.node_list[self.Ed_cols[id]]))
                y_list.append(
                    self.get_edge_weight(self.Ed_rows[id], self.Ed_cols[id], edge_type, positive=True, weighted=True))

            elif edge_type == UNDIRECTED_EDGE:
                X_list.append((self.node_list[self.Eu_rows[id]], self.node_list[self.Eu_cols[id]]))
                y_list.append(
                    self.get_edge_weight(self.Eu_rows[id], self.Eu_cols[id], edge_type, positive=True, weighted=True))

            elif edge_type == UNDIRECTED_NEG_EDGE:
                X_list.append((self.node_list[self.En_rows[id]], self.node_list[self.En_cols[id]]))
                y_list.append(
                    self.get_edge_weight(self.En_rows[id], self.En_cols[id], edge_type, positive=False, weighted=True))
            elif edge_type == DIRECTED_NEG_EDGE:
                X_list.append((self.node_list[self.Ens_rows[id]], self.node_list[self.Ens_cols[id]]))
                y_list.append(self.get_edge_weight(self.Ens_rows[id], self.Ens_cols[id], edge_type, positive=False,
                                                   weighted=False))

        # assert self.batch_size == len(X_list)
        X_list = np.array(X_list, dtype="O")
        y_list = np.array(y_list).reshape((-1, 1))
        return X_list, y_list

    def get_edge_weight(self, i, j, edge_type, positive, weighted):
        if not weighted:
            if positive:
                return 1
            else:
                return 0

        if edge_type == DIRECTED_EDGE:
            return self.adj_directed[i, j]
        elif edge_type == UNDIRECTED_EDGE:
            return self.adj_undirected[i, j]
        elif edge_type == UNDIRECTED_NEG_EDGE:
            return self.adj_negative[i, j]
        elif edge_type == DIRECTED_NEG_EDGE:
            return 0

    def __getdata__(self, edges_batch):
        'Returns the training data (X, y) tuples given a list of tuple(source_id, target_id, is_directed, edge_weight)'
        X_list = []
        for id, edge_type in edges_batch:
            if edge_type == DIRECTED_EDGE:
                X_list.append((self.Ed_rows[id], self.Ed_cols[id], IS_DIRECTED,
                               self.get_edge_weight(self.Ed_rows[id], self.Ed_cols[id],
                                                    edge_type=edge_type,
                                                    positive=True, weighted=self.weighted)))

            elif edge_type == UNDIRECTED_EDGE:
                X_list.append((self.Eu_rows[id], self.Eu_cols[id], IS_UNDIRECTED,
                               self.get_edge_weight(self.Eu_rows[id], self.Eu_cols[id],
                                                    edge_type=edge_type,
                                                    positive=True, weighted=self.weighted)))
            elif edge_type == UNDIRECTED_NEG_EDGE:
                X_list.append((self.En_rows[id], self.En_cols[id], IS_UNDIRECTED,
                               self.get_edge_weight(self.En_rows[id], self.En_cols[id],
                                                    edge_type=edge_type,
                                                    positive=False, weighted=self.weighted
                                                    )))
            elif edge_type == DIRECTED_NEG_EDGE:
                X_list.append((self.Ens_rows[id], self.Ens_cols[id], IS_DIRECTED,
                               self.get_edge_weight(self.Ens_rows[id], self.Ens_cols[id],
                                                    edge_type=edge_type,
                                                    positive=False, weighted=self.weighted
                                                    )
                               ))  # E_ij of negative edges should be 0

        # assert self.batch_size == len(X_list)
        X_list = np.array(X_list, dtype="O")

        X = {}
        X["input_seq_i"] = self.get_sequences([self.node_list[node_id] for node_id in X_list[:, 0].tolist()],
                                              variable_length=False)
        X["input_seq_j"] = self.get_sequences([self.node_list[node_id] for node_id in X_list[:, 1].tolist()],
                                              variable_length=False)
        X["is_directed"] = np.expand_dims(X_list[:, 2], axis=-1)

        y = np.expand_dims(X_list[:, 3].astype(np.float32), axis=-1)

        return X, y

    def load_data(self, return_sequence_data=False, batch_size=None):
        # Returns the y_true labels. Note: run this before running .`on_epoch_end`() since it may reindex the samples
        X = []
        y = []
        for i in range(self.__len__()):
            X_i, y_i = self.get_training_edges(i)
            X.append(X_i)
            y.append(y_i)

        if batch_size:
            X = X[0: batch_size]
            y = y[0: batch_size]

        X = np.vstack(X)
        y = np.vstack(y)
        X = np.array(X, dtype="O")

        if return_sequence_data:
            X_seq = {}
            X[:, 0] = [self.node_list.index(node) for node in X[:, 0].tolist()]
            X[:, 1] = [self.node_list.index(node) for node in X[:, 1].tolist()]
            X_seq["input_seq_i"] = self.get_sequences([self.node_list[node_id] for node_id in X[:, 0].tolist()],
                                                      variable_length=False)
            X_seq["input_seq_j"] = self.get_sequences([self.node_list[node_id] for node_id in X[:, 1].tolist()],
                                                      variable_length=False)
            X_seq["is_directed"] = np.expand_dims(X[:, 2], axis=-1)
            X = X_seq

        return X, y

    def update_negative_samples(self):
        sample_indices = np.random.choice(self.Ens_rows_all.shape[0], self.Ens_count, replace=False)
        self.Ens_rows = self.Ens_rows_all[sample_indices]
        self.Ens_cols = self.Ens_cols_all[sample_indices]

    def process_negative_sampling_edges_filtered(self, node_list_A, node_list_B):
        if self.training_network is not None:
            adj_positive = self.adj_directed + self.adj_undirected + self.training_network.get_adjacency_matrix(
                edge_types=["d"],
                node_list=self.node_list)
        else:
            adj_positive = self.adj_directed + self.adj_undirected
        self.Ens_rows_all, self.Ens_cols_all = np.where(adj_positive.todense() == 0)

        # Filter by nodes list
        node_A_ind = [self.node_list.index(node) for node in self.node_list if node in node_list_A]
        node_B_ind = [self.node_list.index(node) for node in self.node_list if node in node_list_B]
        filter_indices = np.where(np.isin(self.Ens_rows_all, node_A_ind) & np.isin(self.Ens_cols_all, node_B_ind))

        self.Ens_rows_all = self.Ens_rows_all[filter_indices]
        self.Ens_cols_all = self.Ens_cols_all[filter_indices]

    def __getitem__(self, training_index):
        # Generate indexes of the batch
        indices = self.indexes[training_index * self.batch_size: (training_index + 1) * self.batch_size]

        # Find list of IDs
        edges_batch = [self.split_index(i) for i in indices]

        # Generate data
        X, y = self.__getdata__(edges_batch)

        return X, y

    def __len__(self):
        return int(np.floor((self.Ed_count + self.Eu_count + self.En_count + self.Ens_count) / self.batch_size))


class SampledPairsGenerator(PairsGenerator):
    def __init__(self, network, variables=None, targets=None, weighted=False, batch_size=1, replace=True, seed=0,
                 verbose=True, **kwargs):
        """

        Args:
            network (HeterogeneousNetwork):
            weighted:
            batch_size:
            neg_sampling_ratio:
            compression_func:
            n_steps: Number of sampling steps each iteration
            directed_proba:
            maxlen:
            padding:
            truncating:
            sequence_to_matrix:
            tokenizer:
            replace:
            seed:
            verbose:
        """
        self.compression_func = compression_func
        self.n_steps = n_steps
        self.neg_sampling_ratio = neg_sampling_ratio
        self.directed_proba = directed_proba
        super(SampledPairsGenerator, self).__init__(network=network, weighted=weighted, batch_size=batch_size,
                                                    replace=replace, seed=seed, verbose=verbose, maxlen=maxlen,
                                                    padding=padding, truncating=truncating,
                                                    sequence_to_matrix=sequence_to_matrix, tokenizer=tokenizer)
        self.process_sampling_table(network)

    def process_sampling_table(self, network):
        graph = nx.compose(network.G, network.G_u)
        self.edge_dict = {}
        self.edge_counts_dict = {}
        self.node_degrees = dict(zip(self.node_list, [0] * len(self.node_list)))

        for node in network.node_list:
            self.edge_dict[node] = {}
            self.edge_counts_dict[node] = {}

            edgelist_bunch = graph.edges(node, data=True)
            self.node_degrees[node] = len(edgelist_bunch)

            for u,v,d in edgelist_bunch:
                if d["type"] in self.edge_dict[node]:
                    self.edge_dict[node][d["type"]].append((u, v, d["type"]))
                else:
                    self.edge_dict[node][d["type"]] = EdgelistSampler([(u, v, d["type"])])

            for edge_type in self.edge_dict[node].keys():
                self.edge_counts_dict[node][edge_type] = len(self.edge_dict[node][edge_type])

        self.node_degrees_list = [self.node_degrees[node] for node in self.node_list]
        self.node_sampling_freq = self.compute_node_sampling_freq(self.node_degrees_list,
                                                                  compression_func=self.compression_func)
        print("# of nodes to sample from (non-zero degree):", np.count_nonzero(self.node_sampling_freq)) if self.verbose else None

    def get_nonzero_nodelist(self):
        """
        Returns a list of nodes that have an associated edge
        :return:
        """
        return [self.node_list[id] for id in self.node_sampling_freq.nonzero()[0]]

    def process_negative_sampling_edges(self):
        # Negative Directed Edges (sampled)
        adj_positive = self.adj_directed + self.adj_undirected
        self.adj_negative_sampled = adj_positive == 0

        self.Ens_count = int(self.Ed_count * self.negative_sampling_ratio) # Used to calculate sampling ratio to sample negative directed edges
        print("Ens_count:", self.Ens_count) if self.verbose else None

    def compute_node_sampling_freq(self, node_degrees, compression_func):
        if compression_func == "sqrt":
            compression = np.sqrt
        elif compression_func == "sqrt3":
            compression = lambda x: x ** (1 / 3)
        elif compression_func == "log":
            compression = lambda x: np.log(1 + x)
        else:
            compression = lambda x: x

        denominator = sum(compression(np.array(node_degrees)))
        return compression(np.array(node_degrees)) / denominator

    def __len__(self):
        return self.n_steps

    def __getitem__(self, item):
        sampling_nodes = np.random.choice(self.node_list, size=self.batch_size, replace=True,
                                          p=self.node_sampling_freq)

        sampled_edges = [self.sample_edge_from_node(node) for node in sampling_nodes]

        X, y = self.__getdata__(sampled_edges)

        return X, y

    def sample_edge_from_node(self, node):
        edge_type = self.sample_edge_type(self.edge_dict[node].keys())

        if edge_type == DIRECTED_NEG_EDGE:
            return self.get_negative_sampled_edges(node)
        else:
            return next(self.edge_dict[node][edge_type])

    def sample_edge_type(self, edge_types):
        if DIRECTED_EDGE in edge_types and UNDIRECTED_EDGE in edge_types:
            edge_type = np.random.choice([DIRECTED_EDGE, UNDIRECTED_EDGE],
                                         p=[self.directed_proba, 1 - self.directed_proba])
        elif DIRECTED_EDGE in edge_types:
            edge_type = DIRECTED_EDGE
        elif UNDIRECTED_EDGE in edge_types:
            edge_type = UNDIRECTED_EDGE
        elif UNDIRECTED_NEG_EDGE in edge_types:
            edge_type = UNDIRECTED_NEG_EDGE
        else:
            raise Exception("No edge type selected in " + str(edge_types))

        if random.random() < (self.negative_sampling_ratio / (1 + self.negative_sampling_ratio)):
            if edge_type == DIRECTED_EDGE:
                edge_type = DIRECTED_NEG_EDGE
            elif edge_type == UNDIRECTED_EDGE and UNDIRECTED_NEG_EDGE in edge_types:
                edge_type = UNDIRECTED_NEG_EDGE

        return edge_type

    def get_negative_sampled_edges(self, node_u):
        node_idx = self.node_list.index(node_u)
        _, col = self.adj_negative_sampled[node_idx].nonzero()
        node_v = self.node_list[np.random.choice(col)]
        return (node_u, node_v, DIRECTED_NEG_EDGE)

    def __getdata__(self, sampled_edges):
        'Returns the training data (X, y) tuples given a list of tuple(source_id, target_id, is_directed, edge_weight)'
        X_list = []
        for u, v, type in sampled_edges:
            if type == DIRECTED_EDGE:
                X_list.append((u, v, IS_DIRECTED,
                               self.get_edge_weight(self.node_list.index(u), self.node_list.index(v),
                                                    type, True, weighted=self.weighted)))
            elif type == UNDIRECTED_EDGE:
                X_list.append((u, v, IS_UNDIRECTED,
                               self.get_edge_weight(self.node_list.index(u), self.node_list.index(v),
                                                    type, True, weighted=self.weighted)))
            elif type == UNDIRECTED_NEG_EDGE:
                X_list.append((u, v, IS_UNDIRECTED,
                               self.get_edge_weight(self.node_list.index(u), self.node_list.index(v),
                                                    type, False, weighted=self.weighted)))
            elif type == DIRECTED_NEG_EDGE:
                X_list.append((u, v, IS_DIRECTED,
                               self.get_edge_weight(self.node_list.index(u), self.node_list.index(v),
                                                    type, False, weighted=self.weighted)))
            else:
                raise Exception("Edge type is wrong:" + u + v + type)

        # assert self.batch_size == len(X_list)
        X_list = np.array(X_list, dtype="O")

        X = {}
        X["input_seq_i"] = self.get_sequences(X_list[:, 0].tolist(), variable_length=False)
        X["input_seq_j"] = self.get_sequences(X_list[:, 1].tolist(), variable_length=False)
        X["is_directed"] = np.expand_dims(X_list[:, 2], axis=-1)

        y = np.expand_dims(X_list[:, 3].astype(np.float32), axis=-1)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch and shuffle'
        self.indexes = np.arange(self.n_steps)
        self.annotations["Transcript sequence"] = self.sample_sequences(self.transcripts_to_sample)
