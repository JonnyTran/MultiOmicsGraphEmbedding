import pandas as pd
import tensorflow as tf
import torch
from torch.utils import data
from torch.utils.data import BatchSampler

from . import DataGenerator, SubgraphGenerator


class NeighborSampler(BatchSampler):
    def __init__(self, dataset: SubgraphGenerator, batch_size: int, drop_last: bool) -> None:
        self.dataset = dataset
        self.batch_size = self.dataset.batch_size

    def __iter__(self):
        return self.dataset.traverse_network(self.batch_size)

    def __len__(self) -> int:
        return self.dataset.n_steps


class SubgraphDataset(SubgraphGenerator, torch.utils.data.Dataset):
    def __init__(self, network, variables: list = None, targets: list = None, batch_size=500, sampling='neighborhood',
                 compression="log", n_steps=100, directed=True, maxlen=1400, padding='post', truncating='post',
                 agg_mode=None, tokenizer=None, replace=True, variable_length=False, seed=0, verbose=True, **kwargs):
        super(SubgraphDataset, self).__init__(network, variables, targets, batch_size, sampling, compression, n_steps,
                                              directed, maxlen,
                                              padding, truncating, agg_mode, tokenizer, replace, variable_length, seed,
                                              verbose, **kwargs)

        self.node_list = pd.Series(self.node_list)

    def __len__(self):
        return len(self.node_list)

    def __getitem__(self, item=None):
        sampled_nodes = self.node_list[item]
        print("sampled_node", sampled_nodes.shape, sampled_nodes)
        X, y, idx_weights = self.__getdata__(sampled_nodes, variable_length=False)
        return X, y, idx_weights.values


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