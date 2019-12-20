from keras.layers import Layer
from keras.optimizers import Adam

from moge.embedding.siamese_graph_embedding import *
from moge.embedding.siamese_triplet_online_embedding import SiameseOnlineTripletGraphEmbedding
from moge.generator.triplet_generator import OnlineTripletGenerator
from moge.network.heterogeneous_network import HeterogeneousNetwork


class OnlineSoftmaxGraphEmbedding(SiameseOnlineTripletGraphEmbedding):

    def __init__(self, d=128, margin=0.2, batch_size=256, lr=0.001, epochs=10, directed_proba=0.5,
                 compression_func="sqrt", negative_sampling_ratio=2.0, max_length=1400, truncating="post", seed=0,
                 verbose=False, conv1_kernel_size=12, conv1_batch_norm=False, max1_pool_size=6, conv2_kernel_size=6,
                 conv2_batch_norm=True, max2_pool_size=3, lstm_unit_size=320, dense1_unit_size=1024,
                 dense2_unit_size=512,
                 directed_distance="euclidean", undirected_distance="euclidean", source_target_dense_layers=True,
                 embedding_normalization=False, **kwargs):
        self.directed_margin = margin
        self.undirected_margin = margin

        super().__init__(d, margin, batch_size, lr, epochs, directed_proba, compression_func, negative_sampling_ratio,
                         max_length, truncating, seed, verbose, conv1_kernel_size, conv1_batch_norm, max1_pool_size,
                         conv2_kernel_size, conv2_batch_norm, max2_pool_size, lstm_unit_size, dense1_unit_size,
                         dense2_unit_size,
                         directed_distance, undirected_distance, source_target_dense_layers, embedding_normalization,
                         **kwargs)

    def custom_recall(self, loss):
        def loss(y_true, y_pred):
            return K.identity(loss)

        return loss

    def build_keras_model(self, multi_gpu=False):
        if multi_gpu:
            device = "/cpu:0"
            allow_soft_placement = True
        else:
            device = "/gpu:0"
            allow_soft_placement = True

        K.clear_session()
        tf.reset_default_graph()
        config = tf.ConfigProto(allow_soft_placement=allow_soft_placement,
                                log_device_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        with tf.device("/gpu:1" if not multi_gpu else device):
            input_seqs = Input(batch_shape=(self.batch_size, None), dtype=tf.int8, name="input_seqs")
            input_indices = Input(batch_shape=(self.batch_size, None), dtype=tf.int8, name="input_indices")
            labels_directed = Input(batch_shape=(self.batch_size, self.batch_size), sparse=True, dtype=tf.float32,
                                    name="labels_directed")
            labels_undirected = Input(batch_shape=(self.batch_size, self.batch_size), sparse=True, dtype=tf.float32,
                                      name="labels_undirected")
            print("labels_directed", labels_directed) if self.verbose else None
            print("labels_undirected", labels_undirected) if self.verbose else None

            # build create_lstm_network to use in each siamese 'leg'
            self.lstm_network = self.create_lstm_network()
            print("lstm_network", self.lstm_network) if self.verbose else None

            # encode each of the inputs into a list of embedding vectors with the conv_lstm_network
            embeddings = self.lstm_network(input_seqs)
            print("embeddings", embeddings) if self.verbose else None

        with tf.device("/gpu:2" if not multi_gpu else device):
            output = OnlineSoftmaxLoss(directed_margin=self.margin, undirected_margin=self.margin,
                                       undirected_weight=self.directed_proba,
                                       directed_distance=self.directed_distance,
                                       undirected_distance=self.undirected_distance)(
                [embeddings, input_indices, labels_directed, labels_undirected])

            print("output", output) if self.verbose else None

            self.siamese_net = Model(inputs=[input_seqs, input_indices, labels_directed, labels_undirected],
                                     outputs=output)

        # Multi-gpu parallelization
        if multi_gpu:
            self.siamese_net = multi_gpu_model(self.siamese_net, gpus=4, cpu_merge=True, cpu_relocation=False)

        # Build tensorboard
        self.tensorboard = TensorBoard(log_dir="logs/{}_{}".format(type(self).__name__, time.strftime('%m-%d_%l-%M%p')),
                                       histogram_freq=0,
                                       write_grads=True, write_graph=False, write_images=True,
                                       batch_size=self.batch_size,
                                       update_freq="epoch", embeddings_freq=0,
                                       embeddings_metadata="logs/metadata.tsv",
                                       embeddings_data=self.generator_val.__getitem__(0)[0],
                                       embeddings_layer_names=["embedding_output_normalized"],
                                       )

        # Compile & train
        self.siamese_net.compile(loss=self.identity_loss,
                                 optimizer=Adam(lr=self.lr, beta_1=0.9, beta_2=0.999),
                                 )
        print("Network total weights:", self.siamese_net.count_params()) if self.verbose else None

    def learn_embedding(self, network: HeterogeneousNetwork, network_val=None, tensorboard=False, histogram_freq=0,
                        multi_gpu=False, subsample=True, n_steps=500, validation_steps=None, edge_f=None,
                        is_weighted=False, no_python=False, rebuild_model=False, seed=0):
        self.generator_train = OnlineTripletGenerator(network=network, batch_size=self.batch_size,
                                                      maxlen=self.max_length, padding='post',
                                                      truncating=self.truncating, replace=True, seed=seed,
                                                      verbose=self.verbose) \
            if not hasattr(self, "generator_train") else self.generator_train
        self.node_list = self.generator_train.node_list

        if network_val is not None:
            self.generator_val = OnlineTripletGenerator(network=network_val, batch_size=self.batch_size,
                                                        maxlen=self.max_length, padding='post', truncating="post",
                                                        replace=True, seed=seed, verbose=self.verbose) \
                if not hasattr(self, "generator_val") else self.generator_val
        else:
            self.generator_val = None

        if not hasattr(self, "siamese_net") or rebuild_model: self.build_keras_model(multi_gpu)

        if histogram_freq > 0:
            self.tensorboard.histogram_freq = histogram_freq

        try:
            self.hist = self.siamese_net.fit_generator(self.generator_train, epochs=self.epochs,
                                                       validation_data=self.generator_val.__getitem__(
                                                           0) if histogram_freq > 0 else self.generator_val,
                                                       validation_steps=validation_steps,
                                                       callbacks=[self.tensorboard] if tensorboard else None,
                                                       use_multiprocessing=True, workers=8)
        except KeyboardInterrupt:
            print("Stop training")
        finally:
            self.save_alpha_layers()


class OnlineSoftmaxLoss(Layer):
    def __init__(self, node_list, undirected_weight=1.0, directed_distance="euclidean", undirected_distance="euclidean",
                 **kwargs):
        super(OnlineSoftmaxLoss, self).__init__(**kwargs)
        self.node_list = node_list
        self.num_nodes = len(self.node_list)
        self.undirected_weight = undirected_weight
        self.directed_distance = directed_distance
        self.undirected_distance = undirected_distance

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        embeddings_shape, labels_directed_shape, labels_undirected_shape = input_shape
        self._d = int(embeddings_shape[-1])
        self.node_embeddings = self.add_weight(name='node_embeddings',
                                               shape=(self.num_nodes, self._d),
                                               initializer='uniform',
                                               trainable=False)
        self.node_bias = self.add_weight(name='node_bias',
                                         shape=(len(self.node_list),),
                                         initializer='zeros',
                                         trainable=True)

        super(OnlineSoftmaxLoss, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return ()

    def call(self, input, **kwargs):
        assert isinstance(input, list), "{}".format("(embeddings, labels_directed, labels_undirected) expected")
        embeddings, labels_directed, labels_undirected = input

        embeddings_s = embeddings[:, 0: int(self._d / 2)]
        node_embeddings_s = self.node_embeddings[:, 0: int(self._d / 2)]
        node_embeddings_t = self.node_embeddings[:, int(self._d / 2): self._d]

        p1_proximities = _pairwise_dot_sigmoid_distances(embeddings_s, node_embeddings_t)

        p2_proximities = _pairwise_dot_sigmoid_distances(embeddings, self.node_embeddings)
        print("p1_proximities", p1_proximities.shape)
        print("p2_proximities", p2_proximities.shape)




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


def _pairwise_dot_sigmoid_distances(embeddings_A, embeddings_B):
    dot_product = tf.matmul(embeddings_A, tf.transpose(embeddings_B))
    sigmoids = tf.sigmoid(dot_product)
    return sigmoids


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p have a positive edge weight > 0.5.
    Args:
        labels: tf.float32 `Sparse Tensor` with shape [batch_size, batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    positive_edges = tf.sparse_retain(labels, tf.greater(labels.values, 0.5))
    positive_mask = tf.sparse.to_dense(tf.SparseTensor(positive_edges.indices,
                                                       tf.ones_like(positive_edges.values, dtype=tf.bool),
                                                       labels.dense_shape),
                                       default_value=False)
    return positive_mask


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have a negative edge weight (e.g. < 0.5).
    Args:
        labels: tf.float32 `Sparse Tensor` with shape [batch_size, batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    negative_edges = tf.sparse_retain(labels, tf.less(labels.values, 0.5))
    negative_mask = tf.sparse.to_dense(tf.SparseTensor(negative_edges.indices,
                                                       tf.ones_like(negative_edges.values, dtype=tf.bool),
                                                       labels.dense_shape),
                                       default_value=False)
    return negative_mask


def batch_hard_triplet_loss(embeddings_B, embeddings_A, labels, margin, squared=False, distance="euclidean"):
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
        :param distance:
    """
    # Get the pairwise distance matrix
    if distance == "euclidean":
        pairwise_dist = _pairwise_distances(embeddings_A, embeddings_B, squared=squared)
    elif distance == "dot_sigmoid":
        pairwise_dist = _pairwise_dot_sigmoid_distances(embeddings_A, embeddings_B)

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
