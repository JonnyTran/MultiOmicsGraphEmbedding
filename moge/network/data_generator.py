import keras
import numpy as np

from scipy.linalg import triu

from moge.network.heterogeneous_network import HeterogeneousNetwork

class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, network: HeterogeneousNetwork,
                 get_training_data=False,
                 batch_size=1, dim=(None, 4), negative_samples=5,
                 shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.negative_samples = negative_samples
        self.network = network
        self.shuffle = shuffle

        self.adj_directed = network.get_adjacency_matrix(edge_type="d", node_list=list_IDs,
                                                         get_training_data=get_training_data)
        self.adj_undirected = network.get_adjacency_matrix(edge_type="u", node_list=list_IDs,
                                                           get_training_data=get_training_data)

        self.Ed_rows, self.Ed_cols = self.adj_directed.nonzero()  # getting the list of non-zero edges from the Sparse Numpy matrix
        self.Ed_count = len(self.Ed_rows)
        self.Eu_rows, self.Eu_cols = triu(self.adj_undirected,
                                          k=1).nonzero()  # only get non-zero edges from upper triangle of the adjacency matrix
        self.Eu_count = len(self.Eu_rows)

        self.node_list = list_IDs

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor((self.Ed_count + self.Eu_count) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.node_list[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.node_list))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def sample_one_negative_sample(self):
        pass

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = {}
        X["input_seq_i"] = np.empty((self.batch_size, *self.dim))
        X["input_seq_j"] = np.empty((self.batch_size, *self.dim))

        # X =
        y = np.empty((self.batch_size), dtype=int)



        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def seq_to_array(self, seq_str):
        pass
