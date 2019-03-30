import random
from collections import Generator
from collections import OrderedDict

import keras
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from scipy.sparse import triu

from moge.network.heterogeneous_network import HeterogeneousNetwork

IS_UNDIRECTED = 0
IS_DIRECTED = 1
DIRECTED_NEG_EDGE_TYPE = 'd_n'
UNDIRECTED_NEG_EDGE_TYPE = 'u_n'
UNDIRECTED_EDGE_TYPE = 'u'
DIRECTED_EDGE_TYPE = 'd'



class DataGenerator(keras.utils.Sequence):

    def __init__(self, network: HeterogeneousNetwork, weighted=False,
                 batch_size=1, negative_sampling_ratio=3,
                 maxlen=1400, padding='post', truncating='post', sequence_to_matrix=False,
                 shuffle=True, seed=0, verbose=True):
        """
        This class is a data generator for Siamese net Keras models. It generates a sample batch for SGD solvers, where
        each sample in the batch is a uniformly sampled edge of all edge types (negative & positive). The label (y) of
        positive edges have an edge of 1.0, and negative have edge weight of 0.0. The features (x) of each sample is a
        pair of nodes' RNA sequence input.

        :param network: A HeterogeneousNetwork containing a MultiOmicsData
        :param batch_size: Sample batch size at each iteration
        :param dim: Dimensionality of the sample input
        :param negative_sampling_ratio: Ratio of negative edges to positive edges to sample from directed edges
        :param maxlen: pad all RNA sequence strings to this length
        :param padding: ['post', 'pre', None]
        :param sequence_to_matrix: [True, False]
        :param truncating: ['post', 'pre', 'random']. If 'random', then 'post' or 'pre' truncating is chosen randomly for each sequence at each iteration
        :param shuffle:
        :param seed:
        """
        self.batch_size = batch_size
        self.weighted = weighted
        self.negative_sampling_ratio = negative_sampling_ratio
        self.network = network
        self.node_list = network.node_list
        self.node_list = list(OrderedDict.fromkeys(self.node_list))  # Remove duplicates
        self.shuffle = shuffle
        self.padding = padding
        self.maxlen = maxlen
        self.truncating = truncating
        self.seed = seed
        self.sequence_to_matrix = sequence_to_matrix
        self.verbose = verbose
        np.random.seed(seed)

        self.genes_info = network.genes_info
        self.transcripts_to_sample = network.genes_info["Transcript sequence"].copy()

        self.process_training_edges_data()
        self.process_negative_sampling_edges()
        self.on_epoch_end()
        self.process_sequence_tokenizer()


    def process_sequence_tokenizer(self):
        self.tokenizer = Tokenizer(char_level=True, lower=False)
        self.tokenizer.fit_on_texts(self.genes_info.loc[self.node_list, "Transcript sequence"])
        self.genes_info["Transcript length"] = self.genes_info["Transcript sequence"].apply(
            lambda x: len(x) if type(x) == str else None)
        print("word index:", self.tokenizer.word_index) if self.verbose else None

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
        adj_positive = self.adj_directed + self.adj_undirected
        self.Ens_rows_all, self.Ens_cols_all = np.where(adj_positive.todense() == 0)
        self.Ens_count = int(self.Ed_count * self.negative_sampling_ratio)
        print("Ens_count:", self.Ens_count) if self.verbose else None

    def process_negative_sampling_edges_filtered(self, node_list_A, node_list_B):
        # All Negative Directed Edges (non-positive edges)
        adj_positive = self.adj_directed + self.adj_undirected
        self.Ens_rows_all, self.Ens_cols_all = np.where(adj_positive.todense() == 0)

        # Filter by nodes list
        nodes_A = [n for n in self.node_list if n in node_list_A]
        nodes_B = [n for n in self.node_list if n in node_list_B]
        filter_indices = []
        node_A_ind = {self.node_list.index(node) for node in nodes_A}
        node_B_ind = {self.node_list.index(node) for node in nodes_B}
        for i in range(len(self.Ens_rows_all)):
            if self.Ens_rows_all[i] in node_A_ind and self.Ens_cols_all[i] in node_B_ind:
                filter_indices.append(i)

        self.Ens_rows_all = self.Ens_rows_all[filter_indices]
        self.Ens_cols_all = self.Ens_cols_all[filter_indices]

    def update_negative_samples(self):
        sample_indices = np.random.choice(self.Ens_rows_all.shape[0], self.Ens_count, replace=False)
        self.Ens_rows = self.Ens_rows_all[sample_indices]
        self.Ens_cols = self.Ens_cols_all[sample_indices]

    def reload_directed_edges_data(self, edge_types=["d"], databases=None, node_list=None, node_list_B=None):
        self.adj_directed = self.network.get_adjacency_matrix(edge_types=edge_types, node_list=self.node_list,
                                                              databases=databases)
        self.Ed_rows, self.Ed_cols = self.adj_directed.nonzero()  # getting the list of non-zero edges from the Sparse Numpy matrix
        self.Ed_count = len(self.Ed_rows)

        if node_list is not None and node_list_B is not None:
            self.process_negative_sampling_edges_filtered(node_list, node_list_B)

        self.Ens_count = int(self.Ed_count * self.negative_sampling_ratio)
        print("Ed_count:", self.Ed_count, ", Eu_count:", self.Eu_count, ", En_count:",
              self.En_count, ", Ens_count", self.Ens_count) if self.verbose else None
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch and shuffle'
        self.update_negative_samples()
        self.genes_info["Transcript sequence"] = self.sample_sequences(self.transcripts_to_sample)

        self.indexes = np.arange(self.Ed_count + self.Eu_count + self.En_count + self.Ens_count)

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def sample_sequences(self, sequences):
        return sequences.apply(lambda x: random.choice(x) if type(x) is list else x)

    def split_index(self, index):
        'Choose the corresponding edge type data depending on the index number'

        # Index belonging to undirected edges
        if index < self.Ed_count:
            return index, DIRECTED_EDGE_TYPE

        # Index belonging to undirected edges
        elif self.Ed_count <= index and index < (self.Ed_count + self.Eu_count):
            return index - self.Ed_count, UNDIRECTED_EDGE_TYPE

        # index belonging to negative edges
        elif (self.Ed_count + self.Eu_count) <= index and index < (self.Ed_count + self.Eu_count + self.En_count):
            return index - (self.Ed_count + self.Eu_count), UNDIRECTED_NEG_EDGE_TYPE

        # Index belonging to directed negative sampled edges
        elif (self.Ed_count + self.Eu_count + self.En_count) <= index:
            return index - (self.Ed_count + self.Eu_count + self.En_count), DIRECTED_NEG_EDGE_TYPE
        else:
            raise Exception("Index out of range. Value:" + index)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor((self.Ed_count + self.Eu_count + self.En_count + self.Ens_count) / self.batch_size))

    def __getitem__(self, training_index):
        # Generate indexes of the batch
        indices = self.indexes[training_index * self.batch_size: (training_index + 1) * self.batch_size]

        # Find list of IDs
        edges_batch = [self.split_index(i) for i in indices]

        # Generate data
        X, y = self.__data_generation(edges_batch)

        return X, y

    def __data_generation(self, edges_batch):
        'Returns the training data (X, y) tuples given a list of tuple(source_id, target_id, is_directed, edge_weight)'
        X_list = []
        for id, edge_type in edges_batch:
            if edge_type == DIRECTED_EDGE_TYPE:
                X_list.append((self.Ed_rows[id], self.Ed_cols[id], IS_DIRECTED,
                               self.get_edge_weight(self.Ed_rows[id], self.Ed_cols[id],
                                                    edge_type=edge_type,
                                                    positive=True, weighted=self.weighted)))

            elif edge_type == UNDIRECTED_EDGE_TYPE:
                X_list.append((self.Eu_rows[id], self.Eu_cols[id], IS_UNDIRECTED,
                               self.get_edge_weight(self.Eu_rows[id], self.Eu_cols[id],
                                                    edge_type=edge_type,
                                                    positive=True, weighted=self.weighted)))
            elif edge_type == UNDIRECTED_NEG_EDGE_TYPE:
                X_list.append((self.En_rows[id], self.En_cols[id], IS_UNDIRECTED,
                               self.get_edge_weight(self.En_rows[id], self.En_cols[id],
                                                    edge_type=edge_type,
                                                    positive=False, weighted=self.weighted
                                                    )))
            elif edge_type == DIRECTED_NEG_EDGE_TYPE:
                X_list.append((self.Ens_rows[id], self.Ens_cols[id], IS_DIRECTED,
                               self.get_edge_weight(self.Ens_rows[id], self.Ens_cols[id],
                                                    edge_type=edge_type,
                                                    positive=False, weighted=self.weighted
                                                    )
                               ))  # E_ij of negative edges should be 0

        # assert self.batch_size == len(X_list)
        X_list = np.array(X_list, dtype="O")

        X = {}
        X["input_seq_i"] = self.get_sequence_data([self.node_list[node_id] for node_id in X_list[:, 0].tolist()],
                                                  variable_length=False)
        X["input_seq_j"] = self.get_sequence_data([self.node_list[node_id] for node_id in X_list[:, 1].tolist()],
                                                  variable_length=False)
        X["is_directed"] = np.expand_dims(X_list[:,2], axis=-1)

        y = np.expand_dims(X_list[:, 3].astype(np.float32), axis=-1)

        return X, y

    def get_edge_weight(self, i, j, edge_type, positive, weighted):
        if not weighted:
            if positive:
                return 1
            else:
                return 0

        if edge_type == DIRECTED_EDGE_TYPE:
            return self.adj_directed[i, j]
        elif edge_type == UNDIRECTED_EDGE_TYPE:
            return self.adj_undirected[i, j]
        elif edge_type == UNDIRECTED_NEG_EDGE_TYPE:
            return self.adj_negative[i, j]
        elif edge_type == DIRECTED_NEG_EDGE_TYPE:
            return 0

    def make_dataset(self, return_sequence_data=False, batch_size=None):
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
            X_seq["input_seq_i"] = self.get_sequence_data([self.node_list[node_id] for node_id in X[:, 0].tolist()],
                                                          variable_length=False)
            X_seq["input_seq_j"] = self.get_sequence_data([self.node_list[node_id] for node_id in X[:, 1].tolist()],
                                                          variable_length=False)
            X_seq["is_directed"] = np.expand_dims(X[:, 2], axis=-1)
            X = X_seq

        return X, y

    def get_training_edges(self, training_index):
        """
        Generate training edges (for right now only works with directed edges)
        :param training_index:
        :return:
        """
        # Generate indexes of the batch
        indices = self.indexes[training_index * self.batch_size: (training_index + 1) * self.batch_size]

        # Find list of IDs
        edges_batch = [self.split_index(i) for i in indices]

        X_list = []
        y_list = []
        for id, edge_type in edges_batch:
            if edge_type == DIRECTED_EDGE_TYPE:
                X_list.append((self.node_list[self.Ed_rows[id]], self.node_list[self.Ed_cols[id]], IS_DIRECTED))
                y_list.append(
                    self.get_edge_weight(self.Ed_rows[id], self.Ed_cols[id], edge_type, positive=True, weighted=False))

            elif edge_type == UNDIRECTED_EDGE_TYPE:
                X_list.append((self.node_list[self.Eu_rows[id]], self.node_list[self.Eu_cols[id]], IS_UNDIRECTED))
                y_list.append(
                    self.get_edge_weight(self.Eu_rows[id], self.Eu_cols[id], edge_type, positive=True, weighted=False))

            elif edge_type == UNDIRECTED_NEG_EDGE_TYPE:
                X_list.append((self.node_list[self.En_rows[id]], self.node_list[self.En_cols[id]], IS_UNDIRECTED))
                y_list.append(
                    self.get_edge_weight(self.En_rows[id], self.En_cols[id], edge_type, positive=False, weighted=False))
            elif edge_type == DIRECTED_NEG_EDGE_TYPE:
                X_list.append((self.node_list[self.Ens_rows[id]], self.node_list[self.Ens_cols[id]], IS_DIRECTED))
                y_list.append(self.get_edge_weight(self.Ens_rows[id], self.Ens_cols[id], edge_type, positive=False,
                                                   weighted=False))

        # assert self.batch_size == len(X_list)
        X_list = np.array(X_list, dtype="O")
        y_list = np.array(y_list).reshape((-1, 1))
        return X_list, y_list

    def get_sequence_data(self, node_list, variable_length=False, minlen=None):
        """
        Returns an ndarray of shape (batch_size, sequence length, n_words) given a list of node ids
        (indexing from self.node_list)
        :param node_list: a list of node names to fetch transcript sequences
        :param variable_length: returns a list of sequences with different timestep length
        :param minlen: pad all sequences with length lower than this minlen
        """
        if not variable_length:
            padded_encoded_sequences = self.encode_texts(self.genes_info.loc[node_list, "Transcript sequence"],
                                                         maxlen=self.maxlen)
        else:
            padded_encoded_sequences = [
                self.encode_texts([self.genes_info.loc[node, "Transcript sequence"]], minlen=minlen)
                for node in
                node_list]

        return padded_encoded_sequences

    def encode_texts(self, texts, maxlen=None, minlen=None):
        """
        Returns a one-hot-vector for a string of RNA transcript sequence
        :param texts: [str | list(str)]
        :param maxlen: Set length to maximum length
        :param single: Set to True if texts is not a list (i.e. only a single node name string).
        :return:
        """
        # integer encode
        encoded = self.tokenizer.texts_to_sequences(texts)

        batch_maxlen = max([len(x) for x in encoded])
        if batch_maxlen < self.maxlen:
            maxlen = batch_maxlen

        if minlen and len(texts) == 1 and len(texts[0]) < minlen:
            maxlen = minlen

        # pad encoded sequences
        encoded = pad_sequences(encoded, maxlen=maxlen, padding=self.padding,
                                truncating=np.random.choice(
                                    ["post", "pre"]) if self.truncating == "random" else self.truncating,
                                dtype="int8")

        if self.sequence_to_matrix:
            encoded_expanded = np.expand_dims(encoded, axis=-1)

            return np.array([self.tokenizer.sequences_to_matrix(s) for s in encoded_expanded])
        else:
            return encoded




class SampledDataGenerator(DataGenerator):
    def __init__(self, network: HeterogeneousNetwork, weighted=False,
                 batch_size=1, directed_proba=0.5, negative_sampling_ratio=3, n_steps=500, compression_func="log",
                 maxlen=1400, padding='post', truncating='post', sequence_to_matrix=False,
                 shuffle=True, seed=0, verbose=True):
        self.compression_func = compression_func
        self.n_steps = n_steps
        self.directed_proba = directed_proba
        print("Using SampledDataGenerator") if verbose else None
        super().__init__(network, weighted,
                         batch_size, negative_sampling_ratio,
                         maxlen, padding, truncating, sequence_to_matrix,
                         shuffle, seed, verbose)
        self.process_sampling_table(network)

    def process_sampling_table(self, network):
        node_list = network.node_list
        node_list = list(OrderedDict.fromkeys(node_list))

        graph = network.G.subgraph(nodes=node_list)

        self.edge_dict = {}
        self.edge_counts_dict = {}
        self.node_degrees = {}
        for node in network.node_list:
            self.edge_dict[node] = {}
            self.edge_counts_dict[node] = {}

            edgelist_bunch = graph.edges(node, data=True)
            self.node_degrees[node] = len(edgelist_bunch)

            for u,v,d in edgelist_bunch:
                if d["type"] in self.edge_dict[node]:
                    self.edge_dict[node][d["type"]].append((u, v, d["type"]))
                else:
                    self.edge_dict[node][d["type"]] = SampleEdgelistGenerator([(u, v, d["type"])])

            for edge_type in self.edge_dict[node].keys():
                self.edge_counts_dict[node][edge_type] = len(self.edge_dict[node][edge_type])


        self.node_degrees_list = [self.node_degrees[node] for node in node_list]
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

        X, y = self.__data_generation(sampled_edges)

        return X, y

    def sample_edge_from_node(self, node):
        edge_type = self.sample_edge_type(self.edge_dict[node].keys())

        if edge_type == DIRECTED_NEG_EDGE_TYPE:
            return self.get_negative_sampled_edges(node)
        else:
            return next(self.edge_dict[node][edge_type])

    def sample_edge_type(self, edge_types):
        if DIRECTED_EDGE_TYPE in edge_types and UNDIRECTED_EDGE_TYPE in edge_types:
            edge_type = np.random.choice([DIRECTED_EDGE_TYPE, UNDIRECTED_EDGE_TYPE], p=[self.directed_proba, 1-self.directed_proba])
        elif DIRECTED_EDGE_TYPE in edge_types:
            edge_type = DIRECTED_EDGE_TYPE
        elif UNDIRECTED_EDGE_TYPE in edge_types:
            edge_type = UNDIRECTED_EDGE_TYPE
        elif UNDIRECTED_NEG_EDGE_TYPE in edge_types:
            edge_type = UNDIRECTED_NEG_EDGE_TYPE
        else:
            raise Exception("No edge type selected in " + str(edge_types))

        if random.random() < (self.negative_sampling_ratio / (1 + self.negative_sampling_ratio)):
            if edge_type == DIRECTED_EDGE_TYPE:
                edge_type = DIRECTED_NEG_EDGE_TYPE
            elif edge_type == UNDIRECTED_EDGE_TYPE and UNDIRECTED_NEG_EDGE_TYPE in edge_types:
                edge_type = UNDIRECTED_NEG_EDGE_TYPE

        return edge_type

    def get_negative_sampled_edges(self, node_u):
        node_idx = self.node_list.index(node_u)
        _, col = self.adj_negative_sampled[node_idx].nonzero()
        node_v = self.node_list[np.random.choice(col)]
        return (node_u, node_v, DIRECTED_NEG_EDGE_TYPE)

    def __data_generation(self, sampled_edges):
        'Returns the training data (X, y) tuples given a list of tuple(source_id, target_id, is_directed, edge_weight)'
        X_list = []
        for u,v,type in sampled_edges:
            if type == DIRECTED_EDGE_TYPE:
                X_list.append((u, v, IS_DIRECTED,
                               self.get_edge_weight(self.node_list.index(u), self.node_list.index(v),
                                                    type, True, weighted=self.weighted)))
            elif type == UNDIRECTED_EDGE_TYPE:
                X_list.append((u, v, IS_UNDIRECTED,
                               self.get_edge_weight(self.node_list.index(u), self.node_list.index(v),
                                                    type, True, weighted=self.weighted)))
            elif type == UNDIRECTED_NEG_EDGE_TYPE:
                X_list.append((u, v, IS_UNDIRECTED,
                               self.get_edge_weight(self.node_list.index(u), self.node_list.index(v),
                                                    type, False, weighted=self.weighted)))
            elif type == DIRECTED_NEG_EDGE_TYPE:
                X_list.append((u, v, IS_DIRECTED,
                               self.get_edge_weight(self.node_list.index(u), self.node_list.index(v),
                                                    type, False, weighted=self.weighted)))
            else:
                raise Exception("Edge type is wrong:" + u + v + type)

        # assert self.batch_size == len(X_list)
        X_list = np.array(X_list, dtype="O")

        X = {}
        X["input_seq_i"] = self.get_sequence_data(X_list[:, 0].tolist(), variable_length=False)
        X["input_seq_j"] = self.get_sequence_data(X_list[:, 1].tolist(), variable_length=False)
        X["is_directed"] = np.expand_dims(X_list[:,2], axis=-1)

        y = np.expand_dims(X_list[:, 3].astype(np.float32), axis=-1)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch and shuffle'
        self.indexes = np.arange(self.n_steps)
        self.genes_info["Transcript sequence"] = self.sample_sequences(self.transcripts_to_sample)


class SampleEdgelistGenerator(Generator):
    def __init__(self, edgelist:list):
        """
        This class is used to perform sampling without replacement from a node's edgelist
        :param edgelist:
        """
        assert type(edgelist) is list
        self.edgelist = edgelist
        self.sampled_idx = list(np.random.choice(range(len(self.edgelist)), size=len(self.edgelist), replace=False))

    def send(self, ignored_arg=None):
        while True:
            if len(self.sampled_idx) > 0:
                return self.edgelist[self.sampled_idx.pop()]
            else:
                self.sampled_idx = list(
                    np.random.choice(range(len(self.edgelist)), size=len(self.edgelist), replace=False))

    def throw(self, type=None, value=None, traceback=None):
        raise StopIteration

    def append(self, item):
        self.edgelist.append(item)

    def __len__(self):
        return len(self.edgelist)




def main():
    path = "/Users/jonny/Desktop/PycharmProjects/MultiOmicsGraphEmbedding/data/miRNA-mRNA_network.pickle"
    import pickle

    with open(path, "rb") as file:
        network = pickle.load(file)
        file.close()

    network.node_to_modality = {}
    for modality in network.modalities:
        for gene in network.multi_omics_data[modality].get_genes_list():
            network.node_to_modality[gene] = modality

    generator = DataGenerator(network.node_list, network)
    print(generator.__getitem__(1))


if __name__ == "__main__":
    main()
