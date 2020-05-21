import numpy as np
import tensorflow as tf
import torch
from torch.utils import data

from . import DataGenerator


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, generator):
        self.generator = generator

    def __len__(self):
        return len(self.generator.node_list)

    def __getitem__(self, item=None):
        sampled_nodes = self.generator.traverse_network(batch_size=self.generator.batch_size)
        X, y, sample_weights = self.generator.__getdata__(sampled_nodes, variable_length=False)

        X = {k: np.expand_dims(v, 0) for k, v in X.items()}
        y = np.expand_dims(y, 0)
        sample_weights = np.expand_dims(sample_weights, 0)
        return X, y, sample_weights


class TFDataset(tf.data.Dataset):
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