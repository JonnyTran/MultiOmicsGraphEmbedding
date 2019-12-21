import numpy as np

from .sampled_generator import SampledDataGenerator
from collections import OrderedDict

class SubgraphGenerator(SampledDataGenerator):
    def __init__(self, network, variables=None, targets=None, weighted=False, batch_size=500,
                 compression_func="log", n_steps=100, directed_proba=1.0,
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
        sampled_nodes = np.random.choice(self.node_list, size=self.batch_size, replace=False,
                                         p=self.node_sampling_freq)
        X, y = self.__getdata__(sampled_nodes)

        return X, y

    def __getdata__(self, sampled_nodes):
        sampled_nodes = self.annotations.loc[sampled_nodes, self.variables + self.targets].dropna().index.tolist()
        while len(sampled_nodes) < self.batch_size:
            add_nodes = np.random.choice(self.node_list, size=self.batch_size - len(sampled_nodes), replace=False,
                                         p=self.node_sampling_freq).tolist()
            sampled_nodes.append(add_nodes)
            print(type(sampled_nodes))
            sampled_nodes = list(OrderedDict.fromkeys(sampled_nodes))
            sampled_nodes = self.annotations.loc[sampled_nodes, self.variables + self.targets].dropna().index.tolist()

        X = {}
        X["input_seqs"] = self.get_sequence_data(sampled_nodes, variable_length=False)
        X["labels_directed"] = self.network.get_graph_laplacian(edge_types=["d"], node_list=sampled_nodes)

        for variable in self.variables:
            labels_vector = self.annotations.loc[sampled_nodes, variable]
            if labels_vector.dtypes == np.object:
                if labels_vector.str.contains("|").any():
                    labels_vector = labels_vector.str.split("|")
            else:
                labels_vector = labels_vector.to_numpy().reshape(-1, 1)
            X[variable] = self.network.feature_transformer[variable].transform(labels_vector)

        if len(self.targets) == 1:
            targets_vector = self.annotations.loc[sampled_nodes, self.targets[0]]
            if targets_vector.dtypes == np.object:
                if targets_vector.str.contains("|").any():
                    targets_vector = targets_vector.str.split("|")
            else:
                targets_vector = targets_vector.to_numpy().reshape(-1, 1)
            y = self.network.feature_transformer[self.targets[0]].transform(targets_vector)
        else:
            y = {}
            for target in self.targets:
                targets_vector = self.annotations.loc[sampled_nodes, target]
                if targets_vector.dtypes == np.object:
                    if targets_vector.str.contains("|").any():
                        targets_vector = targets_vector.str.split("|")
                else:
                    targets_vector = targets_vector.to_numpy().reshape(-1, 1)
                y[target] = self.network.feature_transformer[target].transform(targets_vector)

        return X, y


    def load_data(self, return_sequence_data=False, batch_size=None):
        pass
