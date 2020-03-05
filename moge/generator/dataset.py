import tensorflow as tf
from tensorflow.keras.utils import OrderedEnqueuer

from . import SubgraphGenerator, DataGenerator


class Dataset(tf.data.Dataset):
    # OUTPUT: (steps, timings, counters)
    OUTPUT_TYPES = (tf.dtypes.string, tf.dtypes.float32, tf.dtypes.int32)
    OUTPUT_SHAPES = (tf.TensorShape([None, None]),
                     tf.TensorShape([None, None]),
                     tf.TensorShape([None, None]))

    def __new__(cls, generator: DataGenerator):
        """
        A tf.data wrapper for keras.utils.Sequence generator
        >>> dataset = SubgraphGenerator(network,)
        >>> strategy = tf.distribute.MirroredStrategy()
        >>> train_dist_dataset = strategy.experimental_distribute_dataset(dataset)

        :param generator: a keras.utils.Sequence generator.
        """
        return tf.data.Dataset.from_generator(
            cls.generator,
            output_types=generator.get_output_types(),
            output_shapes=generator.get_output_shapes(),
            args=(generator)
        )

    def generator(train_generator):
        multi_enqueuer = OrderedEnqueuer(train_generator, use_multiprocessing=True)
        multi_enqueuer.start(workers=10, max_queue_size=10)
        while True:
            batch_xs, batch_ys, dset_index = next(multi_enqueuer.get())  # I have three outputs
            yield batch_xs, batch_ys, dset_index
