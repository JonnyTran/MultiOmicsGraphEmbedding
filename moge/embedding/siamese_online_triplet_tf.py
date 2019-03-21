import tensorflow as tf

from moge.embedding.siamese_triplet_online_embedding import SiameseTripletGraphEmbedding


class SiameseOnlineTripletModel(tf.keras.models.Model, SiameseTripletGraphEmbedding):
    """
    This is a tensorflow implementation of the SiameseTripletGraphEmbedding
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_lstm_network(self):
        """ Base network to be shared (eq. to feature extraction).
        """
        return

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)
