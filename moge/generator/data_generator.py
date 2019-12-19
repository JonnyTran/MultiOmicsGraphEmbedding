import random
from collections import OrderedDict

import keras
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


class SequenceTokenizer():
    def __init__(self, tokenizer=None) -> None:
        if tokenizer is None:
            self.tokenizer = Tokenizer(char_level=True, lower=False)
            self.tokenizer.fit_on_texts(self.genes_info.loc[self.node_list, "Transcript sequence"])
            self.genes_info["Transcript length"] = self.genes_info["Transcript sequence"].apply(
                lambda x: len(x) if type(x) == str else None)
            print("word index:", self.tokenizer.word_index) if self.verbose else None
        else:
            self.tokenizer = tokenizer

    def sample_sequences(self, sequences):
        return sequences.apply(lambda x: random.choice(x) if type(x) is list else x)

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


class DataGenerator(keras.utils.Sequence, SequenceTokenizer):

    def __init__(self, network, weighted=False,
                 batch_size=1, negative_sampling_ratio=3,
                 maxlen=1400, padding='post', truncating='post', tokenizer=None, sequence_to_matrix=False,
                 shuffle=True, seed=0, verbose=True, training_network=None):
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
        self.shuffle = shuffle
        self.padding = padding
        self.maxlen = maxlen
        self.truncating = truncating
        self.seed = seed
        self.sequence_to_matrix = sequence_to_matrix
        self.verbose = verbose
        self.training_network = training_network
        np.random.seed(seed)

        self.genes_info = network.genes_info
        self.transcripts_to_sample = network.genes_info["Transcript sequence"].copy()
        self.node_list = self.genes_info[self.genes_info["Transcript sequence"].notnull()].index.tolist()
        self.node_list = list(OrderedDict.fromkeys(self.node_list))  # Remove duplicates

        self.on_epoch_end()
        super(DataGenerator, self).__init__(tokenizer=tokenizer)

    def on_epoch_end(self):
        'Updates indexes after each epoch and shuffle'
        raise NotImplementedError()

    def __len__(self):
        'Denotes the number of batches per epoch'
        raise NotImplementedError()

    def __getitem__(self, training_index):
        raise NotImplementedError()

    def __data_generation(self, edges_batch):
        raise NotImplementedError()

    def load_data(self, return_sequence_data=False, batch_size=None):
        """
        Returns X, y
        Args:
            return_sequence_data (bool):
            batch_size:
        """
        raise NotImplementedError()
