import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.backend import set_session
from keras.layers import Lambda
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import radius_neighbors_graph

from moge.embedding.siamese_graph_embedding import SiameseGraphEmbedding, sigmoid, softmax
from moge.generator import SampledTripletDataGenerator
from moge.network.heterogeneous_network import HeterogeneousNetwork


class SiameseTripletGraphEmbedding(SiameseGraphEmbedding):
    def __init__(self, d=128, margin=0.2, batch_size=2048, lr=0.001, epochs=10, directed_proba=0.5, weighted=True,
                 compression_func="sqrt", negative_sampling_ratio=2.0, max_length=1400, truncating="post", seed=0,
                 verbose=False, conv1_kernel_size=12, conv1_batch_norm=False, max1_pool_size=6, conv2_kernel_size=6,
                 conv2_batch_norm=True, max2_pool_size=3,
                 lstm_unit_size=320, dense1_unit_size=1024, dense2_unit_size=512,
                 directed_distance="euclidean", undirected_distance="euclidean", source_target_dense_layers=True,
                 embedding_normalization=False,
                 **kwargs):
        super().__init__(d, margin, batch_size, lr, epochs, directed_proba, weighted, compression_func,
                         negative_sampling_ratio,
                         max_length, truncating, seed, verbose, conv1_kernel_size, conv1_batch_norm, max1_pool_size,
                         conv2_kernel_size, conv2_batch_norm,
                         max2_pool_size, lstm_unit_size, dense1_unit_size, dense2_unit_size,
                         directed_distance, undirected_distance, source_target_dense_layers,
                         embedding_normalization,
                         **kwargs)

    def identity_loss(self, y_true, y_pred):
        return K.mean(y_pred - 0 * y_true)

    def triplet_loss(self, inputs):
        encoded_i, encoded_j, encoded_k, is_directed = inputs

        positive_distance = Lambda(self.st_euclidean_distance, name="lambda_positive_distances")(
            [encoded_i, encoded_j, is_directed])
        negative_distance = Lambda(self.st_euclidean_distance, name="lambda_negative_distances")(
            [encoded_i, encoded_k, is_directed])
        return K.mean(K.maximum(0.0, positive_distance - negative_distance + self.margin))

    def build_keras_model(self, multi_gpu=False):
        if multi_gpu:
            device = "/cpu:0"
            allow_soft_placement = True
        else:
            device = "/gpu:0"
            allow_soft_placement = False

        K.clear_session()
        tf.reset_default_graph()
        config = tf.ConfigProto(allow_soft_placement=allow_soft_placement, log_device_placement=True)
        self.sess = tf.Session(config=config)
        set_session(self.sess)

        with tf.device(device):
            input_seq_i = Input(batch_shape=(self.batch_size, None), name="input_seq_i")
            input_seq_j = Input(batch_shape=(self.batch_size, None), name="input_seq_j")
            input_seq_k = Input(batch_shape=(self.batch_size, None), name="input_seq_k")
            is_directed = Input(batch_shape=(self.batch_size, 1), dtype=tf.int8, name="is_directed")

            # build create_network to use in each siamese 'leg'
            self.lstm_network = self.create_lstm_network()

            # encode each of the two inputs into a vector with the conv_lstm_network
            encoded_i = self.lstm_network(input_seq_i)
            print(encoded_i) if self.verbose else None
            encoded_j = self.lstm_network(input_seq_j)
            print(encoded_j) if self.verbose else None
            encoded_k = self.lstm_network(input_seq_k)
            print(encoded_k) if self.verbose else None

            output = Lambda(self.triplet_loss, name="lambda_triplet_loss_output")(
                [encoded_i, encoded_j, encoded_k, is_directed])

            self.siamese_net = Model(inputs=[input_seq_i, input_seq_j, input_seq_k, is_directed], outputs=output)

        # Multi-gpu parallelization
        if multi_gpu:
            self.siamese_net = multi_gpu_model(self.siamese_net, gpus=4, cpu_merge=True, cpu_relocation=False)

        # Compile & train
        self.siamese_net.compile(loss=self.identity_loss,  # binary_crossentropy, cross_entropy, contrastive_loss
                                 optimizer=Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=0.1),
                                 )
        print("Network total weights:", self.siamese_net.count_params()) if self.verbose else None

    def learn_embedding(self, network: HeterogeneousNetwork, network_val=None, multi_gpu=False,
                        subsample=True, n_steps=500, validation_steps=None, tensorboard=True, histogram_freq=0,
                        early_stopping=False,
                        edge_f=None, is_weighted=False, no_python=False, rebuild_model=False, seed=0, **kwargs):

        generator_train = self.get_training_data_generator(network, n_steps, seed)

        if network_val is not None:
            self.generator_val = SampledTripletDataGenerator(network=network_val, weighted=self.weighted,
                                                             batch_size=self.batch_size, maxlen=self.max_length,
                                                             padding='post', truncating="post",
                                                             tokenizer=generator_train.tokenizer, replace=True,
                                                             seed=seed, verbose=self.verbose) \
                if not hasattr(self, "generator_val") else self.generator_val
        else:
            self.generator_val = None

        assert generator_train.tokenizer.word_index == self.generator_val.tokenizer.word_index
        if not hasattr(self, "siamese_net") or rebuild_model: self.build_keras_model(multi_gpu)

        try:
            print(self.log_dir)
            self.hist = self.siamese_net.fit_generator(generator_train, epochs=self.epochs,
                                                       validation_data=self.generator_val,
                                                       validation_steps=validation_steps,
                                                       callbacks=self.get_callbacks(early_stopping, tensorboard,
                                                                                    histogram_freq),
                                                       use_multiprocessing=True, workers=8, **kwargs)
        except KeyboardInterrupt:
            print("Stop training")
        finally:
            self.save_network_weights()

    def get_training_data_generator(self, network, n_steps=250, seed=0):
        if not hasattr(self, "generator_train"):
            self.generator_train = SampledTripletDataGenerator(network=network, weighted=self.weighted,
                                                               batch_size=self.batch_size, maxlen=self.max_length,
                                                               padding='post', truncating=self.truncating, replace=True,
                                                               seed=seed, verbose=self.verbose)
        else:
            return self.generator_train
        self.node_list = self.generator_train.node_list
        return self.generator_train

    def get_reconstructed_adj(self, beta=2.0, X=None, node_l=None, node_l_b=None, edge_type="d", interpolate=False):
        """
        :param X:
        :param node_l: list of node names
        :param edge_type:
        :return:
        """
        if hasattr(self, "reconstructed_adj") and edge_type == "d":
            adj = self.reconstructed_adj
        else:
            embs = self.get_embeddings()
            assert len(self.node_list) == embs.shape[0]

            adj = self._pairwise_similarity(embs, edge_type)

        if interpolate:
            adj = np.interp(adj, (adj.min(), adj.max()), (0, 1))
        if (node_l is None or node_l == self.node_list) and node_l_b is None:
            if edge_type == "d": self.reconstructed_adj = adj  # Cache reconstructed_adj to memory for faster recall
            return adj
        elif set(node_l) < set(self.node_list) or node_l_b is not None:
            return self._select_adj_indices(adj, node_l, node_l_b)
        elif not (set(node_l) < set(self.node_list)):
            raise Exception("A node in node_l is not in self.node_list.")

    def _pairwise_similarity(self, embeddings, edge_type="d"):
        if edge_type == 'd':
            embeddings_X = embeddings[:, 0:int(self._d / 2)]
            embeddings_Y = embeddings[:, int(self._d / 2):self._d]

            if self.directed_distance == "euclidean_ball":
                embeddings_stacked = np.vstack([embeddings_X, embeddings_Y])
                adj = radius_neighbors_graph(embeddings_stacked, radius=self.margin, n_jobs=-2)
                adj = adj[0:embeddings_X.shape[0], :][:, embeddings_X.shape[0]:]
                print("radius_neighbors_graph")

            elif self.directed_distance == "euclidean":
                adj = pairwise_distances(X=embeddings_X,
                                         Y=embeddings_Y,
                                         metric="euclidean", n_jobs=-2)

                # Get node-specific adaptive threshold
                # adj = self.transform_adj_adaptive_threshold(adj, margin=0)
                # adj = self.transform_adj_beta_exp(adj, edge_types="d", sample_negative=self.negative_sampling_ratio)
                adj = np.exp(-2.0 * adj)
                print("Euclidean dist")

            elif self.directed_distance == "cosine":
                adj = pairwise_distances(X=embeddings_X,
                                         Y=embeddings_Y,
                                         metric="cosine", n_jobs=-2)
                print("Cosine similarity")

            elif self.directed_distance == "dot_sigmoid":
                adj = np.matmul(embeddings_X, embeddings_Y.T)
                adj = sigmoid(adj)
                print("Dot product & sigmoid")
            elif self.directed_distance == "dot_softmax":
                adj = np.matmul(embeddings_X, embeddings_Y.T)
                adj = softmax(adj)
                print("Dot product & softmax")

        elif edge_type == 'u':
            if self.undirected_distance == "euclidean_ball":
                adj = radius_neighbors_graph(embeddings, radius=self.margin, n_jobs=-2)

            elif self.undirected_distance == "euclidean":
                adj = pairwise_distances(X=embeddings,
                                         metric="euclidean", n_jobs=-2)
                # adj = np.exp(-2.0 * adj)
                adj = self.transform_adj_beta_exp(adj, edge_types=["u", "u_n"], sample_negative=False)
                # adj = self.transform_adj_adaptive_threshold(adj, margin=self.margin/2)
                print("Euclidean dist")

            elif self.undirected_distance == "cosine":
                adj = pairwise_distances(X=embeddings,
                                         metric="cosine", n_jobs=-2)

            elif self.undirected_distance == "dot_sigmoid":
                adj = np.matmul(embeddings, embeddings.T)
                adj = sigmoid(adj)
            elif self.undirected_distance == "dot_softmax":
                adj = np.matmul(embeddings, embeddings.T)
                adj = softmax(adj)
        else:
            raise Exception("Unsupported edge_type", edge_type)
        return adj

    def transform_adj_adaptive_threshold(self, adj_pred, margin=0.2, edge_types="d"):
        print("adaptive threshold")
        adj_true = self.generator_train.network.get_adjacency_matrix(edge_types=edge_types,
                                                                     node_list=self.node_list)
        self.distance_threshold = self.get_adaptive_threshold(adj_pred, adj_true, margin)
        print("distance_threshold", self.distance_threshold)
        predicted_adj = np.zeros(adj_pred.shape)
        for node_id in range(predicted_adj.shape[0]):
            predicted_adj[node_id, :] = (adj_pred[node_id, :] < self.distance_threshold).astype(float)
        adj_pred = predicted_adj
        return adj_pred

    def get_adaptive_threshold(self, adj_pred, adj_true, margin):
        distance_threshold = np.zeros((len(self.node_list),))
        for nonzero_node_id in np.unique(adj_true.nonzero()[0]):
            _, nonzero_node_cols = adj_true[nonzero_node_id].nonzero()
            positive_distances = adj_pred[nonzero_node_id, nonzero_node_cols]
            distance_threshold[nonzero_node_id] = np.min(positive_distances)
        median_threshold = np.min(distance_threshold[distance_threshold > 0]) + margin / 2
        distance_threshold[distance_threshold == 0] = median_threshold
        return distance_threshold
