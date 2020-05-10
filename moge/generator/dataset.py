import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from torch.utils import data
from torch.utils.data import BatchSampler, Sampler

from . import DataGenerator, SubgraphGenerator


class SubgraphDataset(SubgraphGenerator, torch.utils.data.Dataset):
    def __init__(self, network, variables: list = None, targets: list = None, batch_size=500,
                 traversal='neighborhood', sampling="log", n_steps=100, directed=True,
                 maxlen=1400, padding='post', truncating='post', agg_mode=None, tokenizer=None, replace=True,
                 variable_length=False,
                 seed=0, verbose=True, **kwargs):
        super(SubgraphDataset, self).__init__(network=network,
                                              variables=variables, targets=targets,
                                              batch_size=batch_size,
                                              traversal=traversal, sampling=sampling,
                                              n_steps=n_steps, directed=directed, replace=replace,
                                              maxlen=maxlen, padding=padding, truncating=truncating,
                                              agg_mode=agg_mode,
                                              tokenizer=tokenizer, seed=seed, verbose=verbose, **kwargs)

        self.node_list = pd.Series(self.node_list)

    def __len__(self):
        return len(self.node_list)

    def __getitem__(self, item=None):
        sampled_nodes = self.traverse_network(batch_size=self.batch_size)
        X, y, idx_weights = self.__getdata__(sampled_nodes, variable_length=False)
        X["subnetwork"] = np.expand_dims(X["subnetwork"], 0)
        return X, y, idx_weights


class GeneratorDataset(tf.data.Dataset):
    def __new__(cls, generator: DataGenerator, output_types=None, output_shapes=None):
        """
        A tf.data wrapper for keras.utils.Sequence generator
        >>> generator = DataGenerator()
        >>> dataset = GeneratorDataset(generator)
        >>> strategy = tf.distribute.MirroredStrategy()
        >>> train_dist_dataset = strategy.experimental_distribute_dataset(dataset)

        :param generator: a keras.utils.Sequence generator.
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