from abc import ABCMeta
from itertools import cycle

import numpy as np

from moge.generator import DataGenerator


class SampledDataGenerator(DataGenerator, metaclass=ABCMeta):
    def __init__(self, network, sampling=None, compression_func="log", n_steps=100, directed=True, **kwargs):
        """

        Args:
            compression_func: {"log", "linear", "sqrt", "sqrt3", None}, default: "log". The node degree compression function to calculate the node sampling frequencies.
            n_steps: Number of sampling steps each iteration
            replace: Whether to sample with or without replacement
        """
        self.compression_func = compression_func
        self.sampling = sampling
        self.n_steps = n_steps
        self.directed = directed

        super(SampledDataGenerator, self).__init__(network=network, **kwargs)

        if self.sampling == 'circle':
            self.nodes_circle = cycle(self.node_list)

            self.n_steps = int(np.ceil(len(self.node_list) / self.batch_size))

        self.process_sampling_table(network)

    def process_sampling_table(self, network):
        # graph = nx.compose(network.G, network.G_u)
        # self.edge_dict = {}
        # self.edge_counts_dict = {}

        if self.directed:
            self.node_degrees = {node: degree for node, degree in network.G.degree(self.node_list)}
        else:
            self.node_degrees = {node: degree for node, degree in network.G_u.degree(self.node_list)}

        self.node_degrees_list = [self.node_degrees[node] if node in self.node_degrees else 0 for node in
                                  self.node_list]
        self.node_sampling_freq = self.compute_node_sampling_freq(self.node_degrees_list,
                                                                  compression=self.compression_func)
        print("# of nodes to sample from (non-zero degree):",
              np.count_nonzero(self.node_sampling_freq)) if self.verbose else None
        assert len(self.node_sampling_freq) == len(self.node_list)

    def get_nonzero_nodelist(self):
        """
        Returns a list of nodes that have an associated edge
        :return:
        """
        return [self.node_list[id] for id in self.node_sampling_freq.nonzero()[0]]

    def compute_node_sampling_freq(self, node_degrees, compression):
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
        sampled_nodes = np.random.choice(self.node_list, size=self.batch_size, replace=False,
                                         p=self.node_sampling_freq)

        X, y = self.__getdata__(sampled_nodes)

        return X, y

    def __getdata__(self, sampled_nodes):
        'Returns the training data (X, y) tuples given a list of tuple(source_id, target_id, is_directed, edge_weight)'
        raise NotImplementedError

    def sample_node_list(self, batch_size):
        raise NotImplementedError

    def sample_node_by_freq(self, batch_size):
        sampled_nodes = np.random.choice(self.node_list, size=batch_size, replace=False,
                                         p=self.node_sampling_freq)
        return sampled_nodes
