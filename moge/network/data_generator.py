import keras
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from scipy.linalg import triu as dense_triu
from scipy.sparse import triu
from sklearn.utils import shuffle

from moge.network.heterogeneous_network import HeterogeneousNetwork

UNDIRECTED_EDGE_TYPE = False

DIRECTED_EDGE_TYPE = True

DIRECTED_NEG_EDGE = 'd_n'
UNDIRECTED_NEG_EDGE = 'u_n'
UNDIRECTED_EDGE = 'u'
DIRECTED_EDGE = 'd'


class DataGenerator(keras.utils.Sequence):

    def __init__(self, network: HeterogeneousNetwork,
                 get_training_data=False,
                 batch_size=1, dim=(None, 4), negative_sampling_ratio=3,
                 maxlen=600, padding='post', truncating='post',
                 shuffle=True, seed=0):
        self.dim = dim
        self.batch_size = batch_size
        self.negative_sampling_ratio = negative_sampling_ratio
        self.network = network
        self.shuffle = shuffle
        self.padding = padding
        self.maxlen = maxlen
        self.truncating = truncating
        self.seed = seed
        np.random.seed(seed)

        self.process_genes_info(network)
        self.filter_node_list()
        self.process_sequence_tokenizer()
        self.process_training_edges_data(get_training_data)
        self.process_negative_sampling_edges()

        self.on_epoch_end()

    def process_genes_info(self, network):
        MIR = network.multi_omics_data.MIR.get_genes_info()
        LNC = network.multi_omics_data.LNC.get_genes_info()
        GE = network.multi_omics_data.GE.get_genes_info()
        self.genes_info = pd.concat([GE, MIR, LNC], join="inner", copy=True)
        print("Genes info columns:", self.genes_info.columns)


    def filter_node_list(self):
        self.node_list = self.genes_info[self.genes_info["Transcript sequence"].notnull()].index.tolist()
        print("Number of nodes without seq removed:", len(self.network.node_list) - len(self.node_list))

    def process_sequence_tokenizer(self):
        self.tokenizer = Tokenizer(char_level=True, lower=False)
        self.tokenizer.fit_on_texts(self.genes_info.loc[self.node_list, "Transcript sequence"])
        print("num_words:", self.tokenizer.num_words, self.tokenizer.word_index)

    def process_training_edges_data(self, get_training_data):
        # Directed Edges (regulatory interaction)
        self.adj_directed = self.network.get_adjacency_matrix(edge_type="d", node_list=self.node_list,
                                                              get_training_data=get_training_data)
        self.Ed_rows, self.Ed_cols = self.adj_directed.nonzero()  # getting the list of non-zero edges from the Sparse Numpy matrix
        self.Ed_count = len(self.Ed_rows)

        # Undirected Edges (node similarity)
        self.adj_undirected = self.network.get_adjacency_matrix(edge_type="u", node_list=self.node_list,
                                                                get_training_data=get_training_data)
        self.Eu_rows, self.Eu_cols = triu(self.adj_undirected >= 0.9, k=1).nonzero()
        self.Eu_count = len(self.Eu_rows)

        # Negative Edges (true negative edges from node similarity)
        self.adj_negative = self.network.get_adjacency_matrix(edge_type="u_n", node_list=self.node_list,
                                                              get_training_data=get_training_data)
        self.En_rows, self.En_cols = triu(self.adj_negative, k=1).nonzero()
        self.En_count = len(self.En_rows)

        print("Ed_count:", self.Ed_count, ", Eu_count:", self.Eu_count, ", En_count:", self.En_count)

    def process_negative_sampling_edges(self):
        # Negative Edges (for sampling)
        adj_positive = self.adj_directed + self.adj_undirected + self.adj_negative
        self.Ens_rows, self.Ens_cols = np.where(dense_triu(adj_positive.todense() == 0, k=1))
        self.Ens_count = (self.Ed_count + self.Eu_count) * self.negative_sampling_ratio - self.En_count
        self.Ens_count = int(self.Ens_count)
        print("Ens_count:", self.Ens_count)


    def on_epoch_end(self):
        'Updates indexes after each epoch and shuffle'
        # self.update_negative_samples()

        self.indexes = np.arange(self.Ed_count + self.Eu_count + self.En_count + self.Ens_count)

        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            self.Ens_rows, self.Ens_cols = shuffle(self.Ens_rows, self.Ens_cols)

    def split_index(self, index):
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

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor((self.Ed_count + self.Eu_count + self.En_count) / self.batch_size))

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
            if edge_type == DIRECTED_EDGE:
                X_list.append((self.Ed_rows[id], self.Ed_cols[id], DIRECTED_EDGE_TYPE, 1))
                # self.adj_directed[self.Ed_rows[id], self.Ed_cols[id]]))
            elif edge_type == UNDIRECTED_EDGE:
                X_list.append(
                    (self.Eu_rows[id], self.Eu_cols[id], UNDIRECTED_EDGE_TYPE, 1))
                # self.adj_undirected[self.Eu_rows[id], self.Eu_cols[id]]))
            elif edge_type == UNDIRECTED_NEG_EDGE:
                X_list.append(
                    (self.En_rows[id], self.En_cols[id], UNDIRECTED_EDGE_TYPE, 0))
                # self.adj_negative[self.En_rows[id], self.En_cols[id]]))  # E_ij of negative edges should be 0
            elif edge_type == DIRECTED_NEG_EDGE:
                X_list.append(
                    (self.Ens_rows[id], self.Ens_cols[id], DIRECTED_EDGE_TYPE, 0))  # E_ij of negative edges should be 0
                
        # assert self.batch_size == len(X_list)
        X_list = np.array(X_list, dtype="O")

        X = {}
        X["input_seq_j"] = self.get_sequence_data(X_list[:, 0].tolist())
        X["input_seq_i"] = self.get_sequence_data(X_list[:, 1].tolist())
        X["is_directed"] = np.expand_dims(X_list[:,2], axis=-1)

        y = np.expand_dims(X_list[:, 3].astype(np.float32), axis=-1)

        return X, y

    def get_sequence_data(self, node_list_ids):
        """
        Returns an ndarray of shape (batch_size, sequence length, n_words) given a list of node ids
        (indexing from self.node_list)
        """

        node_list = [self.node_list[i] for i in node_list_ids]

        padded_encoded_sequences = self.encode_texts(self.genes_info.loc[node_list, "Transcript sequence"])
        return padded_encoded_sequences

    def encode_texts(self, texts):
        # integer encode
        encoded = self.tokenizer.texts_to_sequences(texts)
        # pad encoded sequences
        padded_seqs = pad_sequences(encoded, maxlen=self.maxlen, padding=self.padding, truncating=self.truncating)
        # Sequence to matrix
        exp_pad_seqs = np.expand_dims(padded_seqs, axis=-1)
        return np.array([self.tokenizer.sequences_to_matrix(s) for s in exp_pad_seqs])



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
