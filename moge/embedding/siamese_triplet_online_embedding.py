from keras.layers import Layer
from keras.losses import binary_crossentropy
from keras.optimizers import Adadelta

from moge.embedding.siamese_graph_embedding import *
from moge.embedding.siamese_triplet_embedding import SiameseTripletGraphEmbedding
from moge.generator.triplet_generator import OnlineTripletGenerator
from moge.network.heterogeneous_network import HeterogeneousNetwork


class SiameseOnlineTripletGraphEmbedding(SiameseTripletGraphEmbedding):
    def __init__(self, d=128, margin=0.2, batch_size=256, lr=0.001, epochs=10, directed_proba=0.5, weighted=True,
                 compression_func="sqrt", negative_sampling_ratio=2.0, max_length=1400, truncating="post", seed=0,
                 verbose=False, conv1_kernel_size=12, conv1_batch_norm=False, max1_pool_size=6, conv2_kernel_size=6,
                 conv2_batch_norm=True, max2_pool_size=3, lstm_unit_size=320, dense1_unit_size=1024,
                 dense2_unit_size=512,
                 directed_distance="euclidean", undirected_distance="euclidean", source_target_dense_layers=True,
                 embedding_normalization=False, **kwargs):
        self.directed_margin = margin
        self.undirected_margin = margin
        print("directed_margin", self.directed_margin, ", undirected_margin", self.undirected_margin)
        assert directed_proba <= 1.0 and directed_proba >= 0, "directed_proba must be in [0, 1.0] range"
        super().__init__(d, margin, batch_size, lr, epochs, directed_proba, weighted, compression_func,
                         negative_sampling_ratio,
                         max_length, truncating, seed, verbose, conv1_kernel_size, conv1_batch_norm, max1_pool_size,
                         conv2_kernel_size, conv2_batch_norm, max2_pool_size, lstm_unit_size, dense1_unit_size,
                         dense2_unit_size,
                         directed_distance, undirected_distance, source_target_dense_layers, embedding_normalization,
                         **kwargs)

    def custom_recall(self, inputs):
        pairwise_distances, labels = inputs
        y_pred = tf.gather_nd(pairwise_distances, labels.indices)
        y_true = labels.values

        def _recall(_y_true, _y_pred):
            return recall_d(y_true, y_pred)

        return _recall

    def custom_precision(self, inputs):
        pairwise_distances, labels = inputs
        y_pred = tf.gather_nd(pairwise_distances, labels.indices)
        y_true = labels.values

        def _precision(_y_true, _y_pred):
            return precision_d(y_true, y_pred)

        return _precision

    def pairwise_distances(self, embeddings, directed=True, squared=True):
        embeddings_s = embeddings[:, 0:int(self._d / 2)]
        embeddings_t = embeddings[:, int(self._d / 2):self._d]
        if directed:
            return self._pairwise_euclidean(embeddings_s, embeddings_t, squared)
        else:
            if "euclidean_min" in self.undirected_distance:
                return K.minimum(self._pairwise_euclidean(embeddings_s, embeddings_s, squared),
                                 self._pairwise_euclidean(embeddings_t, embeddings_t, squared))
            elif "euclidean" in self.undirected_distance:
                return self._pairwise_euclidean(embeddings, embeddings, squared)


    def _pairwise_euclidean(self, embeddings_s, embeddings_t, squared=True):
        dot_product = K.dot(embeddings_s, K.transpose(embeddings_t))
        square_norm = tf.diag_part(dot_product)
        distances = K.expand_dims(square_norm, 1) - 2.0 * dot_product + K.expand_dims(square_norm, 0)
        distances = K.maximum(distances, 0.0)
        if not squared:
            mask = tf.to_float(tf.equal(distances, 0.0))
            distances = distances + mask * 1e-16
            distances = tf.sqrt(distances)
            # Correct the epsilon added: set the distances on the mask to be exactly 0.0
            distances = distances * (1.0 - mask)
        return distances

    def batch_hard_triplet_loss(self, inputs):
        pairwise_distance_directed, pairwise_distance_undirected, labels_directed, labels_undirected = inputs
        directed_loss = batch_hard_triplet_loss(pairwise_distance_directed, labels_directed,
                                                margin=self.directed_margin)
        undirected_loss = self.directed_proba * batch_hard_triplet_loss(pairwise_distance_undirected, labels_undirected,
                                                                        margin=self.undirected_margin)
        undirected_loss = K.switch(tf.is_nan(undirected_loss), 0.0, undirected_loss)
        def loss(_y_true, _y_pred):
            return directed_loss + undirected_loss
        return loss

    def batch_contrastive_loss(self, inputs):
        pairwise_distance_directed, pairwise_distance_undirected, labels_directed, labels_undirected = inputs
        y_pred_directed = tf.gather_nd(pairwise_distance_directed, labels_directed.indices)
        y_true_directed = labels_directed.values

        y_pred_undirected = tf.gather_nd(pairwise_distance_undirected, labels_undirected.indices)
        y_true_undirected = labels_undirected.values

        undirected_loss = self.directed_proba * contrastive_loss(y_true_undirected, y_pred_undirected,
                                                                 self.undirected_margin)
        undirected_loss = K.switch(tf.is_nan(undirected_loss), 0.0, undirected_loss)

        def _contrastive_loss(_y_true, _y_pred):
            return contrastive_loss(y_true_directed, y_pred_directed, self.directed_margin) + \
                   undirected_loss

        return _contrastive_loss

    def batch_kl_divergence_loss(self, inputs):
        pairwise_similarity_directed, pairwise_similarity_undirected, labels_directed, labels_undirected = inputs
        y_pred_directed = tf.gather_nd(pairwise_similarity_directed, labels_directed.indices)
        y_true_directed = labels_directed.values

        y_pred_undirected = tf.gather_nd(pairwise_similarity_undirected, labels_undirected.indices)
        y_true_undirected = labels_undirected.values

        undirected_loss = self.directed_proba * binary_crossentropy(y_true_undirected, y_pred_undirected)
        undirected_loss = K.switch(tf.is_nan(undirected_loss), 0.0, undirected_loss)
        def _kl_loss(_y_true, _y_pred):
            return binary_crossentropy(y_true_directed, y_pred_directed) + undirected_loss

        return _kl_loss

    def pairwise_similarity(self, embeddings, directed=True):
        embeddings_s = embeddings[:, 0:int(self._d / 2)]
        embeddings_t = embeddings[:, int(self._d / 2):self._d]
        if directed:
            dot_product = K.dot(embeddings_s, K.transpose(embeddings_t))
            return K.sigmoid(dot_product)
        else:
            dot_product = K.dot(embeddings, K.transpose(embeddings))
            return K.sigmoid(dot_product)

    def build_keras_model(self, multi_gpu=False):
        if multi_gpu:
            device = "/cpu:0"
        else:
            device = "/gpu:0"
        K.clear_session()
        tf.reset_default_graph()

        with tf.device(device):
            input_seqs = Input(batch_shape=(None, None), dtype=tf.int8, name="input_seqs")
            labels_directed = Input(batch_shape=(None, None), sparse=True, dtype=tf.float32,
                                    name="labels_directed")
            labels_undirected = Input(batch_shape=(None, None), sparse=True, dtype=tf.float32,
                                      name="labels_undirected")
            print("labels_directed", labels_directed) if self.verbose else None
            print("labels_undirected", labels_undirected) if self.verbose else None

            # build create_lstm_network to use in each siamese 'leg'
            self.lstm_network = self.create_lstm_network()

            # encode each of the inputs into a list of embedding vectors with the conv_lstm_network
            embeddings = self.lstm_network(input_seqs)
            print("embeddings", embeddings) if self.verbose else None

            directed_pairwise_distances = Lambda(lambda x: self.pairwise_distances(x, directed=True, squared=False),
                                                 name="directed_pairwise_distances")(embeddings)
            undirected_pairwise_distances = Lambda(lambda x: self.pairwise_distances(x, directed=False, squared=False),
                                                   name="undirected_pairwise_distances")(embeddings)
            print("directed_pairwise_distances", directed_pairwise_distances) if self.verbose else None

            # self.triplet_loss = OnlineTripletLoss(directed_margin=self.margin, undirected_margin=self.margin,
            #                                       undirected_weight=self.directed_proba,
            #                                       directed_distance=self.directed_distance,
            #                                       undirected_distance=self.undirected_distance)
            # output = self.triplet_loss([embeddings, labels_directed, labels_undirected])
            # print("output", output) if self.verbose else None

            self.siamese_net = Model(inputs=[input_seqs, labels_directed, labels_undirected], outputs=embeddings)

            # Multi-gpu parallelization
            if multi_gpu:
                self.siamese_net = multi_gpu_model(self.siamese_net, gpus=4, cpu_merge=True, cpu_relocation=False)

            # Compile & train
            self.siamese_net.compile(  # loss=self.identity_loss,
                loss=self.batch_contrastive_loss([directed_pairwise_distances,
                                                  undirected_pairwise_distances,
                                                  labels_directed, labels_undirected]),
                optimizer=Adadelta(),
                metrics=[self.custom_precision([directed_pairwise_distances, labels_directed]),
                         self.custom_recall([directed_pairwise_distances, labels_directed])],
                                     )
            print("Network total weights:", self.siamese_net.count_params()) if self.verbose else None

    def learn_embedding(self, network: HeterogeneousNetwork, network_val=None, tensorboard=True, histogram_freq=0,
                        embeddings=False, early_stopping=False,
                        multi_gpu=False, subsample=True, n_steps=500, validation_steps=None,
                        edge_f=None, is_weighted=False, no_python=False, rebuild_model=False, seed=0,
                        **kwargs):
        generator_train = self.get_training_data_generator(network, n_steps, seed)

        if network_val is not None:
            self.generator_val = OnlineTripletGenerator(network=network_val, weighted=self.weighted,
                                                        batch_size=self.batch_size, maxlen=self.max_length,
                                                        padding='post', truncating="post",
                                                        tokenizer=generator_train.tokenizer, shuffle=True, seed=seed,
                                                        verbose=self.verbose) \
                if not hasattr(self, "generator_val") else self.generator_val
        else:
            self.generator_val = None

        assert generator_train.tokenizer.word_index == self.generator_val.tokenizer.word_index
        if not hasattr(self, "siamese_net") or rebuild_model: self.build_keras_model(multi_gpu)

        try:
            self.hist = self.siamese_net.fit_generator(generator_train, epochs=self.epochs, shuffle=False,
                                                       validation_data=self.generator_val,
                                                       validation_steps=validation_steps,
                                                       callbacks=self.get_callbacks(early_stopping, tensorboard,
                                                                                    histogram_freq, embeddings),
                                                       use_multiprocessing=True, workers=8, **kwargs)
        except KeyboardInterrupt:
            print("Stop training")
        finally:
            self.save_network_weights()

    def get_training_data_generator(self, network, n_steps=250, seed=0):
        self.generator_train = OnlineTripletGenerator(network=network, weighted=self.weighted,
                                                      batch_size=self.batch_size, maxlen=self.max_length,
                                                      padding='post', truncating=self.truncating, shuffle=True,
                                                      seed=seed, verbose=self.verbose) \
            if not hasattr(self, "generator_train") else self.generator_train
        self.node_list = self.generator_train.node_list
        return self.generator_train


class OnlineTripletLoss(Layer):
    def __init__(self, directed_margin=0.2, undirected_margin=0.1, undirected_weight=1.0, directed_distance="euclidean",
                 undirected_distance="euclidean", **kwargs):
        super(OnlineTripletLoss, self).__init__(**kwargs)
        self.output_dim = ()
        self.directed_margin = directed_margin
        self.undirected_margin = undirected_margin
        self.undirected_weight = undirected_weight
        self.directed_distance = directed_distance
        self.undirected_distance = undirected_distance

        hyper_params = {}
        hyper_params.update(kwargs)
        for key in hyper_params.keys():
            self.__setattr__('_%s' % key, hyper_params[key])

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        embeddings_shape, labels_directed_shape, labels_undirected_shape = input_shape
        self._d = int(embeddings_shape[-1])

        super(OnlineTripletLoss, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return ()

    def call(self, input, **kwargs):
        assert isinstance(input, list), "(embeddings, labels_directed, labels_undirected) expected"
        embeddings, labels_directed, labels_undirected = input

        embeddings_s = embeddings[:, 0: int(self._d / 2)]
        embeddings_t = embeddings[:, int(self._d / 2):self._d]

        # Get the pairwise distance matrix
        if self.directed_distance == "euclidean":
            directed_pairwise_dist = _pairwise_distances(embeddings_s, embeddings_t, squared=True)
        elif self.directed_distance == "dot_sigmoid":
            directed_pairwise_dist = 1 - _pairwise_dot_sigmoid_similarity(embeddings_s, embeddings_t)
        elif self.directed_distance == "cosine":
            directed_pairwise_dist = 1 - _pairwise_cosine_similarity(embeddings_s, embeddings_t)

        # Get the pairwise distance matrix
        if self.undirected_distance == "euclidean":
            undirected_pairwise_dist = _pairwise_distances(embeddings, embeddings, squared=True)
        elif self.undirected_distance == "euclidean_min":
            undirected_pairwise_dist = tf.minimum(_pairwise_distances(embeddings_s, embeddings_s, squared=False),
                                                  _pairwise_distances(embeddings_t, embeddings_t, squared=False))
        elif self.undirected_distance == "dot_sigmoid":
            undirected_pairwise_dist = 1 - _pairwise_dot_sigmoid_similarity(embeddings, embeddings)
        elif self.undirected_distance == "cosine":
            undirected_pairwise_dist = 1 - _pairwise_cosine_similarity(embeddings, embeddings)

        directed_loss = batch_hard_triplet_loss(directed_pairwise_dist, labels=labels_directed,
                                                margin=self.directed_margin)
        if self.undirected_weight > 0.0:
            undirected_loss = batch_hard_triplet_loss(undirected_pairwise_dist, labels=labels_undirected,
                                                      margin=self.undirected_margin)
            undirected_loss = K.switch(tf.is_nan(undirected_loss), 0.0, undirected_loss)
            return tf.add(directed_loss, self.undirected_weight * undirected_loss)
        else:
            return directed_loss


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
    dot_product = tf.matmul(embeddings_A, embeddings_B, transpose_b=True)

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


def _pairwise_dot_sigmoid_similarity(embeddings_A, embeddings_B):
    dot_product = tf.matmul(embeddings_A, embeddings_B, transpose_b=True)
    sigmoids = tf.sigmoid(dot_product)
    return sigmoids


def _pairwise_cosine_similarity(embeddings_A, embeddings_B):
    normalize_a = tf.nn.l2_normalize(embeddings_A, axis=-1)
    normalize_b = tf.nn.l2_normalize(embeddings_B, axis=-1)
    cosine_similarities = tf.matmul(normalize_a, normalize_b, transpose_b=True)
    return cosine_similarities
# def _pairwise_alpha_similarity(embeddings_A, embeddings_B):

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


def batch_hard_triplet_loss(pairwise_dist, labels, margin):
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
        :param directed:
        :param distance:
    """

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = tf.to_float(mask_anchor_positive)

    # We put to 0 any element where (a, p) is not a connection
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


def frobenius_norm_loss(embeddings_B, embeddings_A, labels: tf.SparseTensor, squared=True, distance="euclidean"):
    """
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
    if distance == "euclidean":
        pairwise_similarity = tf.exp(-_pairwise_distances(embeddings_A, embeddings_B, squared=squared))
    elif distance == "dot_sigmoid":
        pairwise_similarity = _pairwise_dot_sigmoid_similarity(embeddings_A, embeddings_B)
    elif distance == "cosine":
        pairwise_similarity = _pairwise_cosine_similarity(embeddings_A, embeddings_B)

    y_pred = tf.gather_nd(pairwise_similarity, labels.indices)
    frobenius_norm_loss = tf.norm(tf.subtract(y_pred, labels.values), ord=2)

    return frobenius_norm_loss


def batch_constrastive_loss(embeddings_B, embeddings_A, labels: tf.SparseTensor, squared=False, distance="euclidean"):
    if distance == "euclidean":
        pairwise_distances = _pairwise_distances(embeddings_A, embeddings_B, squared=squared)
    elif distance == "dot_sigmoid":
        pairwise_distances = 1 - _pairwise_dot_sigmoid_similarity(embeddings_A, embeddings_B)
    elif distance == "cosine":
        pairwise_distances = 1 - _pairwise_cosine_similarity(embeddings_A, embeddings_B)

    y_pred = tf.gather_nd(pairwise_distances, labels.indices)
    y_true = K.round(labels.values)
    constrastive_loss = contrastive_loss(y_true, y_pred)

    return constrastive_loss
