from abc import ABCMeta
from itertools import cycle

import numpy as np

from .data_generator import DataGenerator


class SampledDataGenerator(DataGenerator, metaclass=ABCMeta):
    def __init__(self, network, traversal=None, traversal_depth=2, sampling="log", n_steps=100, directed=True,
                 **kwargs):
        """

        Args:
            traversal: one of {"node", "neighbor", "bfs", "dfs", "all_slices", "all"}.
            sampling: {"log", "linear", "sqrt", "sqrt3", "cycle", None}, default: "log". The node degree compression function to calculate the node sampling frequencies.
            n_steps: Number of sampling steps each iteration
            replace: Whether to sample with or without replacement
        """
        self.traversal = traversal
        self.sampling = sampling
        self.n_steps = n_steps
        self.directed = directed

        super(SampledDataGenerator, self).__init__(network=network, **kwargs)

        if self.traversal == 'all_slices':
            self.nodes_circle = cycle(self.node_list)
            self.n_steps = int(np.ceil(len(self.node_list) / self.batch_size))
        self.traversal_depth = traversal_depth
        self.process_normalized_node_degree(network)

    def process_normalized_node_degree(self, network):
        if self.directed:
            self.node_degrees = {node: degree for node, degree in network.G.degree(self.node_list)}
        else:
            self.node_degrees = {node: degree for node, degree in network.G_u.degree(self.node_list)}

        self.node_degrees_list = [self.node_degrees[node] if node in self.node_degrees else 0 for node in
                                  self.node_list]
        self.node_sampling_freq = self.normalize_node_degrees(self.node_degrees_list,
                                                              compression=self.sampling)
        print("# of non-zero degree nodes: {}".format(
            np.count_nonzero(self.node_sampling_freq))) if self.verbose else None
        assert len(self.node_sampling_freq) == len(self.node_list)

    def get_connected_nodelist(self):
        """
        Returns a list of nodes that have an associated edge
        :return:
        """
        return [self.node_list[id] for id in self.node_sampling_freq.nonzero()[0]]

    def normalize_node_degrees(self, node_degrees, compression):
        if compression == "sqrt":
            compression_func = np.sqrt
        elif compression == "sqrt3":
            compression_func = lambda x: x ** (1 / 3)
        elif compression == "log":
            compression_func = lambda x: np.log(1 + x)
        elif compression == "linear":
            compression_func = lambda x: np.minimum(x, 1)
        else:
            compression_func = lambda x: x

        denominator = sum(compression_func(np.array(node_degrees)))
        return compression_func(np.array(node_degrees)) / denominator

    def __len__(self):
        return self.n_steps

    def __getitem__(self, item):
        """
        Returns tuple of X, y and sample_weights (optional) as feed inputs to a Keras Model.
        :param item: Not used.
        :return: X, y, sample_weights(optional)
        """
        sampled_nodes = np.random.choice(self.node_list, size=self.batch_size, replace=False,
                                         p=self.node_sampling_freq)

        X, y = self.__getdata__(sampled_nodes)

        return X, y

    def __getdata__(self, sampled_nodes):
        'Returns the training data (X, y) tuples given a list of tuple(source_id, target_id, is_directed, edge_weight)'
        raise NotImplementedError

    def traverse_network(self, batch_size, seed_node=None):
        """
        Sample a traversal on the network.
        :param batch_size (int): size of random walk. Each batch outputs must return the exact batch_size.
        :param seed_node (str, Optional): the starting node. If None, and leaves the traversal method to use the sample_seed_node() method.
        :return: list of nodes in the traversal.
        """
        raise NotImplementedError

    def sample_seed_node(self, size):
        """
        Samples a list of nodes of size `batch_size`, with prior probabilities from `self.node_sampling_freq`.

        :param batch_size (int):
        :return: list of nodes names
        """
        if self.sampling == "cycle":
            return next(self.cycle_random_node_list(size))
        else:
            sampled_nodes = np.random.choice(self.node_list, size=size, replace=False,
                                             p=self.node_sampling_freq)
            return sampled_nodes

    def cycle_random_node_list(self, size):
        random_node_list = np.random.choice(self.node_list, size=len(self.node_sampling_freq.nonzero()[0]),
                                            replace=False, p=self.node_sampling_freq)
        while True:
            if len(random_node_list) >= size:
                yield random_node_list[:size]
                random_node_list = random_node_list[size:]
            else:
                # Resample random_node_list
                random_node_list = np.random.choice(self.node_list, size=len(self.node_sampling_freq.nonzero()[0]),
                                                    replace=False, p=self.node_sampling_freq)
