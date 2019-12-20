import numpy as np

from .sampled_generator import SampledDataGenerator


class SubgraphGenerator(SampledDataGenerator):
    def __init__(self, network, weighted=False, batch_size=1, compression_func="log", n_steps=100, directed_proba=1.0,
                 maxlen=1400, padding='post', truncating='post', sequence_to_matrix=False, tokenizer=None, replace=True,
                 seed=0, verbose=True):
        super(SubgraphGenerator, self).__init__(network=network, weighted=weighted, batch_size=batch_size,
                                                compression_func=compression_func, n_steps=n_steps,
                                                directed_proba=directed_proba,
                                                maxlen=maxlen, padding=padding, truncating=truncating,
                                                sequence_to_matrix=sequence_to_matrix,
                                                tokenizer=tokenizer, replace=replace, seed=seed, verbose=verbose)

    def __getitem__(self, item):
        sampling_nodes = np.random.choice(self.node_list, size=self.batch_size, replace=True,
                                          p=self.node_sampling_freq)

        X, y = self.__getdata__(sampling_nodes)

        return X, y

    def __getdata__(self, sampled_nodes):
        pass

    def __len__(self):
        pass

    def on_epoch_end(self):
        pass

    def load_data(self, return_sequence_data=False, batch_size=None):
        pass
