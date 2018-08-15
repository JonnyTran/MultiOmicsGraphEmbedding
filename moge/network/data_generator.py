import numpy as np
import pandas as pd

from scipy.linalg import triu

import keras

from moge.network.heterogeneous_network import HeterogeneousNetwork

class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, network: HeterogeneousNetwork,
                 get_training_data=False,
                 batch_size=1, dim=(None, 4), negative_sampling_ratio=5,
                 shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        # self.negative_sampling_ratio = negative_sampling_ratio
        self.network = network
        self.shuffle = shuffle
        self.node_list = list_IDs

        self.process_training_edges_data(get_training_data)

        # Negative Edges (for sampling)
        # self.negative_edges = np.argwhere(np.isnan(self.adj_directed + self.adj_undirected + self.adj_negative))

        self.process_genes_info(network)

        self.on_epoch_end()

    def process_genes_info(self, network):
        MIR = network.multi_omics_data.MIR.get_genes_info()
        LNC = network.multi_omics_data.LNC.get_genes_info()
        GE = network.multi_omics_data.GE.get_genes_info()
        self.genes_info = pd.concat([GE, MIR, LNC], join="inner", copy=True)

    def process_training_edges_data(self, get_training_data):
        # Directed Edges (regulatory interaction)
        self.adj_directed = self.network.get_adjacency_matrix(edge_type="d", node_list=self.node_list,
                                                              get_training_data=get_training_data)
        self.Ed_rows, self.Ed_cols = self.adj_directed.nonzero()  # getting the list of non-zero edges from the Sparse Numpy matrix
        self.Ed_count = len(self.Ed_rows)
        # Undirected Edges (node similarity)
        self.adj_undirected = self.network.get_adjacency_matrix(edge_type="u", node_list=self.node_list,
                                                                get_training_data=get_training_data)
        self.Eu_rows, self.Eu_cols = self.adj_undirected.nonzero()  # TODO only get non-zero edges from upper triangle of the adjacency matrix # TODO upper trianglar
        self.Eu_count = len(self.Eu_rows)
        # # Negative Edges (true negative edges from node similarity)
        self.adj_negative = self.network.get_adjacency_matrix(edge_type="u_n", node_list=self.node_list,
                                                              get_training_data=get_training_data)
        self.En_rows, self.En_cols = self.adj_negative.nonzero()  # TODO only get non-zero edges from upper triangle of the adjacency matrix
        self.En_count = len(self.En_rows)

        print("Ed_count", self.Ed_count, "Eu_count", self.Eu_count, "En_count", self.En_count)


    def split_index(self, index):
        if index < self.Ed_count:  # Index belonging to undirected edges
            return index, "d"
        elif self.Ed_count <= index and index < (self.Ed_count + self.Eu_count):  # Index belonging to undirected edges
            return index - self.Ed_count, "u"
        elif index >= (self.Ed_count + self.Eu_count):  # index belonging to negative edges
            return index - (self.Ed_count + self.Eu_count), "u_n"
        else:
            raise Exception("Index out of range. Value:" + index)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor((self.Ed_count + self.Eu_count + self.En_count) / self.batch_size))

    def __getitem__(self, training_index):
        # Generate indexes of the batch
        indices = self.indexes[training_index * self.batch_size: (training_index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.split_index(i) for i in indices]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch and shuffle'
        # self.update_negative_samples()

        self.indexes = np.arange(self.Ed_count + self.Eu_count + self.En_count)

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def update_negative_samples(self):
        self.negative = np.random.shuffle(self.negative)

    def sample_one_negative_sample(self):
        pass

    def __data_generation(self, edges_batch):
        """

        :param list_IDs_temp:
        :return: X : (batch_size, *dim, n_channels)
        """
        #

        X_list = []

        for id, edge_type in edges_batch:
            if edge_type == 'd':
                X_list.append((self.Ed_rows[id],
                               self.Ed_cols[id],
                               True,
                               self.adj_directed[self.Ed_rows[id], self.Ed_cols[id]]))
            elif edge_type == 'u':
                X_list.append(
                    (self.Eu_rows[id],
                     self.Eu_cols[id],
                     False,
                     self.adj_undirected[self.Eu_rows[id], self.Eu_cols[id]]))
            elif edge_type == 'u_n':
                X_list.append(
                    (self.En_rows[id],
                     self.En_cols[id],
                     False,
                     self.adj_negative[self.En_rows[id], self.En_cols[id]]))  # E_ij of negative edges should be 0

        batch_size = len(X_list)

        X = {}
        X["input_seq_i"] = [None, ] * batch_size  # np.empty((batch_size, *self.dim))
        X["input_seq_j"] = [None, ] * batch_size  # np.empty((batch_size, *self.dim))
        X["input_seq_j"] = [None, ] * batch_size  # np.empty((batch_size, *self.dim))

        y = np.empty((batch_size), dtype=np.float32)

        for i, tuple  in enumerate(X_list):
            node_i_id, node_j_id, is_directed, E_ij = tuple
            X["input_seq_i"][i] = self.get_gene_info(self.node_list[node_i_id])
            X["input_seq_j"][i] = self.get_gene_info(self.node_list[node_j_id])
            X["is_directed"][i] = is_directed
            y[i] = E_ij

        return X, y



    def get_gene_info(self, gene_name):
        return self.seq_to_array(self.genes_info.loc[gene_name, "Transcript length"])

    def seq_to_array(self, seq):
        arr = np.zeros((len(seq), 4))
        for i in range(len(seq)):
            if seq[i] == "A":
                arr[i] = np.array([1, 0, 0, 0])
            elif seq[i] == "C":
                arr[i] = np.array([0, 1, 0, 0])
            elif seq[i] == "G":
                arr[i] = np.array([0, 0, 1, 0])
            elif seq[i] == "T":
                arr[i] = np.array([0, 0, 0, 1])
            else:
                arr[i] = np.array([1, 1, 1, 1])


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
