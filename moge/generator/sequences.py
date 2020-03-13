import random

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

SEQUENCE_COL = "Transcript sequence"

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

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            if isinstance(self.annotations, pd.DataFrame):
                self.tokenizer = Tokenizer(char_level=True, lower=False)
                self.tokenizer.fit_on_texts(self.annotations.loc[self.node_list, SEQUENCE_COL])
                print("word index:", self.tokenizer.word_index) if self.verbose else None
            elif isinstance(self.annotations, dict) or isinstance(self.annotations, pd.Series):
                self.tokenizer = {}
                for modality, annotation in self.annotations.items():
                    self.tokenizer[modality] = Tokenizer(char_level=True, lower=False)
                    self.tokenizer[modality].fit_on_texts(
                        annotation.loc[annotation[SEQUENCE_COL].notnull(), SEQUENCE_COL])
                    print("{} word index: {}".format(modality,
                                                     self.tokenizer[modality].word_index)) if self.verbose else None

    def sample_sequences(self, sequences):
        return sequences.apply(lambda x: random.choice(x) if isinstance(x, list) else x)

    def get_sequence_encodings(self, node_list: list, modality=None, variable_length=False, minlen=None):
        """
        Returns an ndarray of shape (batch_size, sequence length, n_words) given a list of node ids
        (indexing from self.node_list)
        :param node_list: a list of node names to fetch transcript sequences
        :param variable_length: returns a list of sequences with different timestep length
        :param minlen: pad all sequences with length lower than this minlen
        """
        if modality is None:
            annotations = self.annotations
        else:
            annotations = self.annotations[modality]

        seqs = self.get_sequences(annotations, node_list)

        if not variable_length:
            try:
                padded_encoded_seqs = self.encode_texts(seqs, maxlen=self.maxlen,
                                                        modality=modality if modality else None, minlen=20)
            except Exception as e:
                print("seqs", seqs.shape, seqs.notnull().sum())
                return (seqs, node_list)
        else:
            padded_encoded_seqs = [
                self.encode_texts([annotations.loc[node, SEQUENCE_COL]], minlen=minlen,
                                  modality=modality if modality else None) \
                for node in node_list]

        return padded_encoded_seqs

    def get_sequences(self, annotation, node_list):
        if set(annotation.index) > set(node_list):
            seqs = annotation.loc[node_list, SEQUENCE_COL]
        else:
            # return dummy string if the annotation doesn't have index
            seqs = pd.Series(node_list).map(lambda x: annotation[SEQUENCE_COL].get(x, "_"))

        return seqs

    def encode_texts(self, texts, modality: str = None, maxlen=None, minlen=None):
        """
        Returns a one-hot-vector for a string of RNA transcript sequence
        :param texts: [str | list(str)]
        :param maxlen: Set length to maximum length
        :param single: Set to True if texts is not a list (i.e. only a single node name string).
        :return:
        """
        # integer encode
        if modality is None:
            tokenizer = self.tokenizer
        else:
            tokenizer = self.tokenizer[modality]

        encoded = tokenizer.texts_to_sequences(texts)

        batch_maxlen = max([len(x) for x in encoded])
        if batch_maxlen < self.maxlen:
            maxlen = batch_maxlen

        if minlen and len(texts[0]) < minlen:
            maxlen = minlen

        # pad encoded sequences
        encoded = pad_sequences(encoded, maxlen=maxlen, padding=self.padding,
                                truncating=np.random.choice(
                                    ["post", "pre"]) if self.truncating == "random" else self.truncating,
                                dtype="int8")

        if not self.sequence_to_matrix:
            return encoded
        else:
            encoded_expanded = np.expand_dims(encoded, axis=-1)
            return np.array([tokenizer.sequences_to_matrix(s) for s in encoded_expanded])
