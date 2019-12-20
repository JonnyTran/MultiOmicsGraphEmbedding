import numpy as np

from .node_classification import ClassificationGenerator
from .sampled_generator import SampledDataGenerator


class SubgraphGenerator(SampledDataGenerator, ClassificationGenerator):
    def __init__(self, network, weighted=False, batch_size=1, compression_func="log", n_steps=100, directed_proba=1.0,
                 maxlen=1400, padding='post', truncating='post', sequence_to_matrix=False, tokenizer=None, replace=True,
                 seed=0, verbose=True, variables=None, targets=None):
        super(SubgraphGenerator, self).__init__(network=network, weighted=weighted, batch_size=batch_size,
                                                compression_func=compression_func, n_steps=n_steps,
                                                directed_proba=directed_proba, replace=replace,
                                                maxlen=maxlen, padding=padding, truncating=truncating,
                                                sequence_to_matrix=sequence_to_matrix,
                                                tokenizer=tokenizer, seed=seed, verbose=verbose,
                                                variables=variables, targets=targets)

    def __getitem__(self, item):
        sampling_nodes = np.random.choice(self.node_list, size=self.batch_size, replace=True,
                                          p=self.node_sampling_freq)

        X, y = self.__getdata__(sampling_nodes)

        return X, y

    def __getdata__(self, sampled_nodes):
        X = {}
        X["input_seqs"] = self.get_sequence_data(sampled_nodes, variable_length=False)
        X["labels_directed"] = self.network.get_adjacency_matrix(edge_types=["d"], node_list=sampled_nodes)

        X.update(self.get_variables(sampled_nodes))
        y = self.get_targets(sampled_nodes)

        return X, y

    def on_epoch_end(self):
        pass

    def load_data(self, return_sequence_data=False, batch_size=None):
        pass
