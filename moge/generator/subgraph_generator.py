import numpy as np
import pandas as pd

from .sampled_generator import SampledDataGenerator


class SubgraphGenerator(SampledDataGenerator):
    def __init__(self, network, variables=None, targets=None, weighted=False, batch_size=1, compression_func="log",
                 n_steps=100, directed_proba=1.0,
                 maxlen=1400, padding='post', truncating='post', sequence_to_matrix=False, tokenizer=None, replace=True,
                 seed=0, verbose=True):
        self.variables = variables
        self.targets = targets
        super(SubgraphGenerator, self).__init__(network=network, weighted=weighted, batch_size=batch_size,
                                                compression_func=compression_func, n_steps=n_steps,
                                                directed_proba=directed_proba, replace=replace,
                                                maxlen=maxlen, padding=padding, truncating=truncating,
                                                sequence_to_matrix=sequence_to_matrix,
                                                tokenizer=tokenizer, seed=seed, verbose=verbose, )

    def __getitem__(self, item):
        print("self.node_sampling_freq", self.node_sampling_freq)
        sampled_nodes = np.random.choice(self.node_list, size=self.batch_size, replace=True,
                                         p=self.node_sampling_freq)
        print("sampling_nodes", sampled_nodes)
        X, y = self.__getdata__(sampled_nodes)

        return X, y

    def __getdata__(self, sampled_nodes):
        X = {}
        X["input_seqs"] = self.get_sequence_data(sampled_nodes, variable_length=False)
        X["labels_directed"] = self.network.get_adjacency_matrix(edge_types=["d"], node_list=sampled_nodes)

        for variable in self.variables:
            X[variable] = pd.get_dummies(self.annotations.loc[sampled_nodes], columns=[variable, ],
                                         dummy_na=True).to_numpy()
        y = pd.get_dummies(self.annotations.loc[sampled_nodes], columns=self.targets,
                           dummy_na=True).to_numpy()

        return X, y


    def load_data(self, return_sequence_data=False, batch_size=None):
        pass
