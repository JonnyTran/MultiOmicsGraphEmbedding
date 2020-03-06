import tensorflow as tf

from . import SubgraphGenerator, DataGenerator


class Dataset(tf.data.Dataset):
    def __new__(cls, generator: DataGenerator):
        """
        A tf.data wrapper for keras.utils.Sequence generator
        >>> dataset = SubgraphGenerator()
        >>> strategy = tf.distribute.MirroredStrategy()
        >>> train_dist_dataset = strategy.experimental_distribute_dataset(dataset)

        :param generator: a keras.utils.Sequence generator.
        """
        cls.generator = generator

        return tf.data.Dataset.from_generator(
            cls.generate,
            output_types=generator.get_output_types(),
            output_shapes=generator.get_output_shapes(),
            args=(generator)
        )

    def generate(generator: DataGenerator):
        while True:
            batch_xs, batch_ys, dset_index = generator.__getitem__()
            yield batch_xs, batch_ys, dset_index
