import numpy as np

from moge.generator.siamese.pairs_generator import DataGenerator


class SampledDataGenerator(DataGenerator):
    def __init__(self, network, weighted=False, batch_size=1,
                 compression_func="log", n_steps=100, directed_proba=1.0,
                 maxlen=1400, padding='post', truncating='post', sequence_to_matrix=False, tokenizer=None,
                 replace=True, seed=0, verbose=True):
        """

        Args:
            n_steps: Number of sampling steps each iteration
            replace: Whether to sample with or without replacement
        """
        self.compression_func = compression_func
        self.n_steps = n_steps
        self.directed_proba = directed_proba
        super(SampledDataGenerator, self).__init__(network=network, weighted=weighted, batch_size=batch_size,
                                                   replace=replace, seed=seed, verbose=verbose,
                                                   maxlen=maxlen, padding=padding, truncating=truncating,
                                                   sequence_to_matrix=sequence_to_matrix, tokenizer=tokenizer
                                                   )
        self.process_sampling_table(network)

    def process_sampling_table(self, network):
        # graph = nx.compose(network.G, network.G_u)
        self.edge_dict = {}
        self.edge_counts_dict = {}
        self.node_degrees = {node: degree for node, degree in network.G.degree(self.node_list)}

        self.node_degrees_list = [self.node_degrees[node] if node in self.node_degrees else 0 for node in
                                  self.node_list]
        self.node_sampling_freq = self.compute_node_sampling_freq(self.node_degrees_list,
                                                                  compression=self.compression_func)
        print("# of nodes to sample from (non-zero degree):",
              np.count_nonzero(self.node_sampling_freq)) if self.verbose else None

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
        else:
            compression_func = lambda x: x

        denominator = sum(compression_func(np.array(node_degrees)))
        return compression_func(np.array(node_degrees)) / denominator

    def __len__(self):
        return self.n_steps

    def __getitem__(self, item):
        sampled_nodes = np.random.choice(self.node_list, size=self.batch_size, replace=self.replace,
                                         p=self.node_sampling_freq)

        X, y = self.__getdata__(sampled_nodes)

        return X, y

    def __getdata__(self, sampled_nodes):
        'Returns the training data (X, y) tuples given a list of tuple(source_id, target_id, is_directed, edge_weight)'
        raise NotImplementedError()

    def on_epoch_end(self):
        'Updates indexes after each epoch and shuffle'
        self.indexes = np.arange(self.n_steps)
        self.annotations["Transcript sequence"] = self.sample_sequences(self.transcripts_to_sample)
