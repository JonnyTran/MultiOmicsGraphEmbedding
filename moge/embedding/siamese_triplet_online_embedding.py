import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.layers import Input, Lambda, Activation, Subtract, Reshape

from moge.embedding.siamese_graph_embedding import *
from moge.embedding.static_graph_embedding import ImportedGraphEmbedding
from moge.evaluation.metrics import accuracy_d, precision_d, recall_d, auc_roc_d, precision, recall, auc_roc
from moge.network.data_generator import DataGenerator, SampledDataGenerator, SampledTripletDataGenerator
from moge.network.heterogeneous_network import HeterogeneousNetwork


class SiameseTripletGraphEmbedding(SiameseTripletGraphEmbedding):

    def __init__(self, d=128, margin=0.2, batch_size=2048, lr=0.001, epochs=10, directed_proba=0.5,
                 compression_func="sqrt", negative_sampling_ratio=2.0, max_length=1400, truncating="post", seed=0,
                 verbose=False, conv1_kernel_size=12, max1_pool_size=6, conv2_kernel_size=6, max2_pool_size=3,
                 lstm_unit_size=320, dense1_unit_size=1024, dense2_unit_size=512, **kwargs):
        super().__init__(d, margin, batch_size, lr, epochs, directed_proba, compression_func, negative_sampling_ratio,
                         max_length, truncating, seed, verbose, conv1_kernel_size, max1_pool_size, conv2_kernel_size,
                         max2_pool_size, lstm_unit_size, dense1_unit_size, dense2_unit_size, **kwargs)

    def build_keras_model(self, multi_gpu=False):
        if multi_gpu:
            device = "/cpu:0"
            allow_soft_placement = True
        else:
            device = "/gpu:0"
            allow_soft_placement = False

        K.clear_session()
        tf.reset_default_graph()
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=allow_soft_placement))

        with tf.device(device):
            input_seqs = Input(batch_shape=(self.batch_size, None), name="input_seqs")
            labels_directed = Input(batch_shape=(self.batch_size, self.batch_size), sparse=True, name="labels_directed")
            labels_undirected = Input(batch_shape=(self.batch_size, self.batch_size), sparse=True,
                                      name="labels_undirected")

            # build create_lstm_network to use in each siamese 'leg'
            self.lstm_network = self.create_lstm_network()

            # encode each of the inputs into a list of embedding vectors with the conv_lstm_network
            input_seqs = self.lstm_network(input_seqs)
            print(input_seqs) if self.verbose else None

            output = OnlineTripletLoss(margin=self.margin)([input_seqs, labels_directed, labels_undirected])

            self.siamese_net = Model(inputs=[input_seqs, labels_directed, labels_undirected], outputs=output)

        # Multi-gpu parallelization
        if multi_gpu:
            self.siamese_net = multi_gpu_model(self.siamese_net, gpus=4, cpu_merge=True, cpu_relocation=False)

        # Compile & train
        self.siamese_net.compile(loss=self.identity_loss,  # binary_crossentropy, cross_entropy, contrastive_loss
                                 optimizer=Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=0.1),
                                 # metrics=[accuracy_d, precision_d, recall_d, auc_roc_d], # constrastive_loss
                                 # metrics=["accuracy", precision, recall], # cross_entropy
                                 )
        print("Network total weights:", self.siamese_net.count_params()) if self.verbose else None

    def learn_embedding(self, network: HeterogeneousNetwork, network_val=None, validation_make_data=False,
                        multi_gpu=False, subsample=True, n_steps=500, validation_steps=None, edge_f=None,
                        is_weighted=False, no_python=False, rebuild_model=False, seed=0):
        super().learn_embedding(network, network_val, validation_make_data, multi_gpu, subsample, n_steps,
                                validation_steps, edge_f, is_weighted, no_python, rebuild_model, seed)


class OnlineTripletLoss(tf.keras.layers.Layer):
    def __init__(self, margin=0.2, undirected_weight=1.0):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.undirected_weight = undirected_weight

    def build(self, input_shape):
        embeddings_shape, labels_directed_shape, labels_undirected_shape = input_shape
        self._d = int(embeddings_shape[-1])
        # self.kernel = self.add_variable("kernel",
        #                                 shape=[int(input_shape[-1]),
        #                                        self.num_outputs])

    def call(self, input):
        embeddings, labels_directed, labels_undirected = input

        embeddings_s = tf.slice(embeddings, [-1, 0],
                                [embeddings.get_shape()[0], int(self._d / 2)])
        embeddings_t = tf.slice(embeddings, [-1, int(self._d / 2)],
                                [embeddings.get_shape()[0], int(self._d / 2)])

        directed_loss = batch_hard_triplet_loss(embeddings_t, embeddings_s, labels_directed, self.margin)
        undirected_loss = batch_hard_triplet_loss(embeddings, embeddings, labels_undirected, self.margin)
        return directed_loss + self.undirected_weight * undirected_loss


def _pairwise_distances(embeddings_A, embeddings_B, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings_A, tf.transpose(embeddings_B))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: tf.float32 `Sparse Tensor` with shape [batch_size, batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    # Combine the two masks
    mask = tf.logical_and(indices_not_equal, labels_equal)

    return mask


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: tf.float `Sparse Tensor` with shape [batch_size, batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_not(labels_equal)

    return mask


def batch_hard_triplet_loss(embeddings_B, embeddings_A, labels, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.
    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings_A, embeddings_B, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = tf.to_float(mask_anchor_positive)

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

    # shape (batch_size, 1)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_positive_dist", tf.reduce_mean(hardest_positive_dist))

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = tf.to_float(mask_anchor_negative)

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_negative_dist", tf.reduce_mean(hardest_negative_dist))

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)

    # Get final mean triplet loss
    triplet_loss = tf.reduce_mean(triplet_loss)

    return triplet_loss
