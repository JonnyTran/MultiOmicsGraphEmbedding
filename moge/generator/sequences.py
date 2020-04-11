import random

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

SEQ_DTYPE = "long"
SEQUENCE_COL = "Transcript sequence"

class SequenceTokenizer():
    def __init__(self, annotations, node_list, padding='post', maxlen=2000, truncating='post', agg_mode=None,
                 tokenizer=None, verbose=False) -> None:
        """
        Handles text tokenizing for DNA/RNA/Protein sequences.

        Args:
            padding: ['post', 'pre', None]
            maxlen (int): pad all RNA sequence strings to this length
            truncating: ['post', 'pre', 'random']. If 'random', then 'post' or 'pre' truncating is chosen randomly for each sequence at each iteration
            agg_mode: one of {"count", "tfidf", "binary", "freq"}, default None. If not None, instead of returning sequence
                encoding, get_sequence_encoding will return an aggregated numpy vector.
            tokenizer: pass an existing tokenizer instead of creating one
        """
        self.maxlen = maxlen
        self.padding = padding
        self.truncating = truncating
        self.agg_mode = agg_mode
        self.annotations = annotations
        self.node_list = node_list
        self.verbose = verbose

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = Tokenizer(char_level=True, lower=False)
            self.tokenizer.fit_on_texts(self.annotations.loc[self.node_list, SEQUENCE_COL])
            print("word index:", self.tokenizer.word_index) if self.verbose else None

    def sample_sequences(self, sequences):
        return sequences.apply(lambda x: random.choice(x) if isinstance(x, list) else x)

    def get_sequence_encodings(self, node_list: list, variable_length=False, minlen=None):
        """
        Returns an ndarray of shape (batch_size, sequence length, n_words) given a list of node ids
        (indexing from self.node_list)
        :param node_list: a list of node names to fetch transcript sequences
        :param variable_length: returns a list of sequences with different timestep length
        :param minlen: pad all sequences with length lower than this minlen
        """
        annotations = self.annotations
        seqs = annotations.loc[node_list, SEQUENCE_COL]

        padded_encoded_seqs = self.encode_texts(seqs, maxlen=self.maxlen, minlen=minlen,
                                                variable_length=variable_length)

        return padded_encoded_seqs

    def encode_texts(self, texts, maxlen=None, minlen=None, variable_length=False):
        """
        Returns a one-hot-vector for a string of RNA transcript sequence
        :param texts: [str | list(str)]
        :param maxlen: Set length to maximum length
        :param single: Set to True if texts is not a list (i.e. only a single node name string).
        :return:
        """
        # integer encode
        encoded = self.tokenizer.texts_to_sequences(texts)

        if variable_length:
            return encoded
        elif self.agg_mode:
            return self.tokenizer.sequences_to_matrix(encoded, mode=self.agg_mode)

        # Pad sequences to the same length
        batch_maxlen = max([len(x) for x in encoded])
        if batch_maxlen < self.maxlen:
            maxlen = batch_maxlen

        if minlen and len(texts[0]) < minlen:
            maxlen = minlen

        # pad encoded sequences
        encoded = pad_sequences(encoded, maxlen=maxlen, padding=self.padding,
                                truncating=np.random.choice(
                                    ["post", "pre"]) if self.truncating == "random" else self.truncating,
                                dtype=SEQ_DTYPE)

        return encoded


class MultiSequenceTokenizer(SequenceTokenizer):
    def __init__(self, annotations, node_list, padding='post', maxlen=2000, truncating='post', agg_mode=None,
                 tokenizer=None) -> None:
        self.padding = padding
        self.maxlen = maxlen
        self.truncating = truncating
        self.agg_mode = agg_mode
        self.annotations_dict = annotations
        self.node_list = node_list

        assert isinstance(self.annotations_dict, dict) or isinstance(self.annotations_dict, pd.Series)

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = {}
            for modality, annotation in self.annotations_dict.items():
                self.tokenizer[modality] = Tokenizer(char_level=True, lower=False)
                self.tokenizer[modality].fit_on_texts(
                    annotation.loc[annotation[SEQUENCE_COL].notnull(), SEQUENCE_COL])
                print(
                    "{} word index: {}".format(modality, self.tokenizer[modality].word_index)) if self.verbose else None

    def get_sequence_encodings(self, node_list: list, modality, variable_length=False, minlen=None):
        """
        Returns an ndarray of shape (batch_size, sequence length, n_words) given a list of node ids
        (indexing from self.node_list)
        :param node_list: a list of node names to fetch transcript sequences
        :param variable_length: returns a list of sequences with different timestep length
        :param minlen: pad all sequences with length lower than this minlen
        """
        annotations = self.annotations_dict[modality]

        seqs = self.fetch_sequences(annotations, node_list)

        try:
            padded_encoded_seqs = self.encode_texts(seqs, modality=modality, maxlen=self.maxlen, minlen=minlen,
                                                    variable_length=variable_length)
        except Exception as e:
            print("seqs", seqs.shape, seqs.notnull().sum())
            raise e

        return padded_encoded_seqs

    def fetch_sequences(self, annotation: pd.DataFrame, node_list: list):
        if set(annotation.index) > set(node_list):
            seqs = annotation.loc[node_list, SEQUENCE_COL]
        else:
            # return dummy string if the annotation doesn't have index
            seqs = pd.Series(node_list).map(lambda x: annotation[SEQUENCE_COL].get(x, ""))

        return seqs

    def encode_texts(self, texts, modality, maxlen=None, minlen=None, variable_length=False):
        """
        Returns a one-hot-vector for a string of RNA transcript sequence
        :param texts: [str | list(str)]
        :param maxlen: Set length to maximum length
        :param single: Set to True if texts is not a list (i.e. only a single node name string).
        :return:
        """
        # integer encode
        tokenizer = self.tokenizer[modality]

        encoded = tokenizer.texts_to_sequences(texts)

        if variable_length:
            return encoded
        elif self.agg_mode:
            return tokenizer.sequences_to_matrix(encoded, mode=self.agg_mode)

        # Pad sequences to the same length
        batch_maxlen = max([len(x) for x in encoded])
        if batch_maxlen < self.maxlen:
            maxlen = batch_maxlen

        if minlen and len(texts[0]) < minlen:
            maxlen = minlen

        # pad encoded sequences
        encoded = pad_sequences(encoded, maxlen=maxlen, padding=self.padding,
                                truncating=np.random.choice(
                                    ["post", "pre"]) if self.truncating == "random" else self.truncating,
                                dtype=SEQ_DTYPE)

        return encoded
