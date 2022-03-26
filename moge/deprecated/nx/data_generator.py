import copy
from collections import OrderedDict

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from moge.graph.sequences import SequenceTokenizer, SEQUENCE_COL
from tensorflow import keras

from moge.graph.hetero import HeteroNetwork
# import moge
from moge.graph.multi_digraph import MultiDigraphNetwork


class DataGenerator(keras.utils.Sequence, SequenceTokenizer):
    def __init__(self, network, variables=None, targets=None, method="GAT", adj_output="dense", sparse_target=False,
                 weighted=False, batch_size=1, replace=True, seed=0,
                 verbose=True, **kwargs):
        """
        This class is a data data for Siamese net Keras models. It generates a sample batch for SGD solvers, where
        each sample in the batch is a uniformly sampled edge of all edge types (negative & positive). The label (y) of
        positive edges have an edge of 1.0, and negative have edge weight of 0.0. The features (x) of each sample is a
        pair of nodes' RNA sequence input.

        :param network: A AttributedNetwork containing a MultiOmics
        :param node_list: optional, default None. Used to explicitly limit processing to a set of nodes.
        :param batch_size: Sample batch size at each iteration
        :param dim: Dimensionality of the sample input
        :param negative_sampling_ratio: Ratio of negative edges to positive edges to sample from directed edges
        :param replace: {True, False}, default True.
        :param seed:
        """
        self.batch_size = batch_size
        self.weighted = weighted
        self.network = copy.deepcopy(network)  # Prevent shared pointers between training/testing
        self.replace = replace

        self.method = method
        self.adj_output = adj_output
        self.sparse_target = sparse_target

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

        if not hasattr(self, "node_list") or self.node_list is None:
            self.node_list = self.network.node_list

        if isinstance(self.network, MultiDigraphNetwork):  # Heterogeneous network
            # Ensure every node must have an associated sequence
            valid_nodes = self.annotations[self.annotations[SEQUENCE_COL].notnull()].index.tolist()
            self.node_list = [node for node in self.node_list if node in valid_nodes]

            # Subgraph to training/testing
            self.network.G = self.network.G.subgraph(nodes=self.node_list).copy()
            self.network.G_u = self.network.G_u.subgraph(nodes=self.node_list).copy()
            print("node_list", len(self.node_list),
                  {"directed": self.network.G.number_of_nodes(), "undirected": self.network.G_u.number_of_nodes()})

        elif isinstance(self.network, HeteroNetwork):  # Multiplex network
            # Check that each node must have sequence data in all layers
            null_nodes = [network.annotations[modality].loc[network.nodes[modality], SEQUENCE_COL][
                              network.annotations[modality].loc[
                                  network.nodes[modality], SEQUENCE_COL].isnull()].index.tolist() for modality in
                          network.node_types]
            null_nodes = [node for nodes in null_nodes for node in nodes]

            self.node_list = [node for node in self.node_list if node not in null_nodes]

            # Subgraph to training/testing
            for key, graph in self.network.networks.items():
                self.network.networks[key] = graph.subgraph(nodes=self.node_list).copy()

            print("node_list", len(self.node_list),
                  {key: graph.number_of_nodes() for key, graph in self.network.networks.items()})
        else:
            raise Exception("Check that `annotations` must be a dict of DataFrame or a DataFrame", self.annotations)

        # Remove duplicates
        self.node_list = list(OrderedDict.fromkeys(self.node_list))

        np.random.seed(seed)
        self.on_epoch_end()
        super(DataGenerator, self).__init__(annotations=network.annotations, node_list=self.node_list, **kwargs)

    def on_epoch_end(self):
        'Updates indexes after each epoch and shuffle'
        if self.n_steps:
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

    def info(self):
        X, y, idx_train = self.__getitem__(0)
        print({k: v.shape if not isinstance(v, list) else (len(v), len(v[0])) for k, v in X.items()},
              {"y_train": y.shape}, idx_train.index[:5])


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

    def process_label(self, targets_vector):
        if not hasattr(self.network, "delimiter"):
            raise Exception(
                "self.network.delimiter doesn't exist. Must run network.process_feature_tranformer(filter_label=targets[0], delimiter='\||, ', min_count=0)")

        delim = self.network.delimiter
        if targets_vector.dtypes == np.object:
            if targets_vector.str.contains(delim, regex=True).any():
                targets_vector = targets_vector.str.split(delim)

            if any(targets_vector.isnull()):
                targets_vector = targets_vector.map(lambda x: x if type(x) == list else [])
            elif all(targets_vector.isnull()):
                targets_vector = targets_vector.to_numpy().reshape(-1, 1)
        else:
            targets_vector = targets_vector.to_numpy().reshape(-1, 1)
        return targets_vector

    def label_sparsify(self, y):
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)

        nz = pd.Series(np.count_nonzero(y, axis=1))
        max_nz = nz.max()
        dfs = []
        for _nz, nzdf in y.groupby(nz, sort=False):
            nz = np.apply_along_axis(lambda r: np.nonzero(r)[0], 1, nzdf)
            mock_result = pd.DataFrame(np.ones(shape=(len(nzdf), max_nz)) - 2, index=nzdf.index).astype(int)

            for i in range(nz.shape[1]):
                mock_result.iloc[:, i] = nz[:, i]
            dfs.append(mock_result)

        result = pd.concat(dfs).sort_index()
        return result

    def label_probability(self, y):
        # Make a probability distribution
        y = (1 / y.sum(axis=1)).reshape(-1, 1) * y
        return y


class GeneratorDataset(torch.utils.data.Dataset):
    def __init__(self, generator):
        self._generator = generator
        self.node_list = self._generator.get_connected_nodelist()
        self.n_steps = self._generator.n_steps

    def __len__(self):
        if self.n_steps is not None:
            return self.n_steps
        else:
            return len(self.node_list)

    def __getitem__(self, item=None):
        # seed_node = self.node_list[item]
        sampled_nodes = self._generator.traverse_network(batch_size=self._generator.batch_size, seed_node=None)
        X, y, sample_weights = self._generator.__getdata__(sampled_nodes, variable_length=False)
        X = {k: np.expand_dims(v, 0) for k, v in X.items()}
        y = np.expand_dims(y, 0)
        sample_weights = np.expand_dims(sample_weights, 0)
        return X, y, sample_weights


class TFDataset(tf.data.Dataset):
    def __new__(cls, generator, output_types=None, output_shapes=None):
        """
        A tf.data wrapper for keras.utils.Sequence data
        >>> data = DataGenerator()
        >>> dataset = GeneratorDataset(data)
        >>> strategy = tf.distribute.MirroredStrategy()
        >>> train_dist_dataset = strategy.experimental_distribute_dataset(dataset)

        :param generator: a keras.utils.Sequence data.
        """

        def generate():
            while True:
                batch_xs, batch_ys, dset_index = generator.__getitem__(0)
                yield batch_xs, batch_ys, dset_index

        queue = tf.keras.utils.GeneratorEnqueuer(generate, use_multiprocessing=True)

        return tf.data.Dataset.from_generator(
            queue.sequence,
            output_types=generator.get_output_types() if output_types is None else output_types,
            output_shapes=generator.get_output_shapes() if output_shapes is None else output_shapes,
        )
