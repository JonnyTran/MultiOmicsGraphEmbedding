import random

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

SEQUENCE = "Transcript sequence"

class SequenceTokenizer():
    def __init__(self, padding='post', maxlen=2000, truncating='post', sequence_to_matrix=None, tokenizer=None) -> None:
        """

        Args:
            padding: ['post', 'pre', None]
            maxlen (int): pad all RNA sequence strings to this length
            truncating: ['post', 'pre', 'random']. If 'random', then 'post' or 'pre' truncating is chosen randomly for each sequence at each iteration
            sequence_to_matrix: a dict(<word in sequence>: <integer code>)
            tokenizer: pass an existing tokenizer instead of creating one
        """
        self.padding = padding
        self.maxlen = maxlen
        self.truncating = truncating
        self.sequence_to_matrix = sequence_to_matrix

        if tokenizer is None:
            self.tokenizer = Tokenizer(char_level=True, lower=False)
            self.tokenizer.fit_on_texts(self.annotations.loc[self.node_list, "Transcript sequence"])
            print("word index:", self.tokenizer.word_index) if self.verbose else None
        else:
            self.tokenizer = tokenizer

    def sample_sequences(self, sequences):
        return sequences.apply(lambda x: random.choice(x) if isinstance(x, list) else x)

    def get_sequence_data(self, node_list, variable_length=False, minlen=None):
        """
        Returns an ndarray of shape (batch_size, sequence length, n_words) given a list of node ids
        (indexing from self.node_list)
        :param node_list: a list of node names to fetch transcript sequences
        :param variable_length: returns a list of sequences with different timestep length
        :param minlen: pad all sequences with length lower than this minlen
        """
        if not variable_length:
            padded_encoded_sequences = self.encode_texts(self.annotations.loc[node_list, "Transcript sequence"],
                                                         maxlen=self.maxlen)
        else:
            padded_encoded_sequences = [
                self.encode_texts([self.annotations.loc[node, "Transcript sequence"]], minlen=minlen)
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
