import tensorflow as tf

from . import DataGenerator


class Dataset(tf.data.Dataset):
    def __new__(cls, generator: DataGenerator):
        """
        A tf.data wrapper for keras.utils.Sequence generator
        >>> generator = DataGenerator()
        >>> dataset = Dataset(generator)
        >>> strategy = tf.distribute.MirroredStrategy()
        >>> train_dist_dataset = strategy.experimental_distribute_dataset(dataset)

        :param generator: a keras.utils.Sequence generator.
        """

        def generate():
            while True:
                batch_xs, batch_ys, dset_index = generator.__getitem__(0)
                yield batch_xs, batch_ys, dset_index

        return tf.data.Dataset.from_generator(
            generate,
            output_types=generator.get_output_types(),
            output_shapes=generator.get_output_shapes(),
        )
