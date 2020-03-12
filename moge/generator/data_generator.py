from collections import OrderedDict

import numpy as np
import pandas as pd
from tensorflow import keras

from moge.generator.sequences import SequenceTokenizer, SEQUENCE_COL


class DataGenerator(keras.utils.Sequence, SequenceTokenizer):
    def __init__(self, network, variables=None, targets=None, weighted=False, batch_size=1,
                 replace=True, seed=0,
                 verbose=True, **kwargs):
        """
        This class is a data generator for Siamese net Keras models. It generates a sample batch for SGD solvers, where
        each sample in the batch is a uniformly sampled edge of all edge types (negative & positive). The label (y) of
        positive edges have an edge of 1.0, and negative have edge weight of 0.0. The features (x) of each sample is a
        pair of nodes' RNA sequence input.

        :param network: A AttributedNetwork containing a MultiOmics
        :param node_list: optional, default None. Pass if explicitly wants to limit processing to a set of nodes.
        :param batch_size: Sample batch size at each iteration
        :param dim: Dimensionality of the sample input
        :param negative_sampling_ratio: Ratio of negative edges to positive edges to sample from directed edges
        :param replace: {True, False}, default True.
        :param seed:
        """
        self.batch_size = batch_size
        self.weighted = weighted
        self.network = network
        self.replace = replace

        self.seed = seed
        self.verbose = verbose

        self.annotations = network.annotations
        if isinstance(network.annotations, pd.DataFrame):
            self.transcripts_to_sample = network.annotations[SEQUENCE_COL].copy()
        else:
            self.transcripts_to_sample = None

        if variables or targets:
            self.variables = variables
            self.targets = targets

        # Initialize node_list
        if "node_list" in kwargs:
            self.node_list = kwargs["node_list"]
            kwargs.pop("node_list")
        else:
            self.node_list = self.network.node_list

        # Ensure every node must have an associated sequence
        if isinstance(self.annotations, pd.DataFrame):
            self.node_list = [node for node in self.node_list if node in self.annotations[
                self.annotations[SEQUENCE_COL].notnull()].index.tolist()]
        elif isinstance(self.annotations, dict) or isinstance(self.annotations, pd.Series):
            # Check that each node must have a sequence in all modalities it's associated with
            self.node_list = [node for node in self.node_list if \
                              all(list(map(lambda modality: bool(self.annotations[modality].loc[node, SEQUENCE_COL]),
                                           network.node_to_modality[node])))]
        else:
            raise Exception("Check that `annotations` must be a dict of DataFrame or a DataFrame", self.annotations)

        # Remove duplicates
        self.node_list = list(OrderedDict.fromkeys(self.node_list))

        np.random.seed(seed)
        self.on_epoch_end()
        super(DataGenerator, self).__init__(**kwargs)

    def on_epoch_end(self):
        'Updates indexes after each epoch and shuffle'
        self.indexes = np.arange(self.n_steps)
        if self.transcripts_to_sample is not None:
            self.annotations[SEQUENCE_COL] = self.sample_sequences(self.transcripts_to_sample)

    def __len__(self):
        'Denotes the number of batches per epoch'
        raise NotImplementedError()

    def __getitem__(self, training_index):
        raise NotImplementedError()

    def __getdata__(self, edges_batch):
        raise NotImplementedError()

    def load_data(self, return_sequence_data=False, batch_size=None):
        """
        Returns X, y
        Args:
            return_sequence_data (bool):
            batch_size:
        """
        raise NotImplementedError()

    def get_output_types(self):
        raise NotImplementedError()

    def get_output_shapes(self):
        raise NotImplementedError()

    def get_node_labels(self, label, node_list):
        if node_list is None:
            node_list = self.node_list

        return self.annotations.loc[node_list, label]

    def get_expressions(self, node_list, modality):
        """
        Get annotation expression associated to node_lists
        :param node_list:
        :param modality:
        :return:
        """
        if modality:
            return self.network.multiomics[modality].get_annotation_expressions().loc[node_list]
        elif modality is None:
            if isinstance(node_list, list):
                node_list = pd.Series(node_list)
            return node_list.map(
                lambda node: self.multiomics[self.node_to_modality[node]].get_annotation_expressions().loc[node])
