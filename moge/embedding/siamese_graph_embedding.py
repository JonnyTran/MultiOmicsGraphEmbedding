import os
import time

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import TensorBoard, EarlyStopping
from keras.layers import Conv2D, Dense, Dropout, Bidirectional, CuDNNLSTM, Embedding
from keras.layers import Dot, MaxPooling1D, Convolution1D, BatchNormalization, Concatenate
from keras.layers import Input, Lambda
from keras.models import Model
from keras.models import load_model
from keras.optimizers import RMSprop
from keras.utils import multi_gpu_model
from sklearn.base import BaseEstimator
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import radius_neighbors_graph

from moge.embedding.static_graph_embedding import ImportedGraphEmbedding
from moge.evaluation.metrics import precision_d, recall_d, precision, recall
from moge.network.edge_generator import DataGenerator, SampledDataGenerator
from moge.network.heterogeneous_network import HeterogeneousNetwork


def contrastive_loss(y_true, y_pred, margin=1.0):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def cross_entropy(y_true, y_pred):
    return -K.sum(y_true * K.log(y_pred)) - K.sum((1-y_true) * K.log(1 - y_pred))

def euclidean_distance(inputs):
    x, y = inputs
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def distance_to_probability(input):
    return K.exp(-input)

def get_abs_diff(inputs):
    x, y = inputs
    return K.abs(x - y)

def get_st_abs_diff(inputs):
    encoded_i, encoded_j = inputs
    dim = encoded_i.shape[-1]
    return K.abs(encoded_i[:, 0:int(dim/2)] - encoded_j[:, int(dim/2):dim])

def abs_diff_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1

def switch(inputs):
    is_directed, directed, undirected = inputs
    return K.switch(is_directed, directed, undirected)

def softmax(X):
    exps = np.exp(X)
    return exps / np.sum(exps, axis=0)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def l1_diff_alpha(u, v, weights):
    l1_diff = np.abs(u - v)
    matmul = np.dot(l1_diff, weights[0]) + weights[1]
    return sigmoid(matmul)



class SiameseGraphEmbedding(ImportedGraphEmbedding, BaseEstimator):
    def __init__(self, d=128, margin=0.2, batch_size=2048, lr=0.001, epochs=10, directed_proba=0.5, weighted=False,
                 compression_func="sqrt", negative_sampling_ratio=2.0,
                 max_length=1400, truncating="post", seed=0, verbose=False,
                 conv1_kernel_size=12, conv1_batch_norm=False, max1_pool_size=6, conv2_kernel_size=6,
                 conv2_batch_norm=True,
                 max2_pool_size=3, lstm_unit_size=320, dense1_unit_size=1024, dense2_unit_size=512,
                 directed_distance="euclidean", undirected_distance="euclidean",
                 source_target_dense_layers=True, embedding_normalization=False, subsample=False,
                 **kwargs):
        super().__init__(d)

        self._d = d
        assert self._d % 2 == 0, "Embedding dimension (d) must be an even integer"
        self.batch_size = batch_size
        self.margin = margin
        self.lr = lr
        self.epochs = epochs
        self.compression_func = compression_func
        self.directed_proba = directed_proba
        self.weighted = weighted
        self.negative_sampling_ratio = negative_sampling_ratio
        self.max_length = max_length
        self.truncating = truncating
        self.seed = seed
        self.verbose = verbose
        self.subsample = subsample

        self.conv1_kernel_size = conv1_kernel_size
        self.conv1_batch_norm = conv1_batch_norm
        self.max1_pool_size = max1_pool_size
        self.conv2_kernel_size = conv2_kernel_size
        self.conv2_batch_norm = conv2_batch_norm
        self.max2_pool_size = max2_pool_size
        self.lstm_unit_size = lstm_unit_size
        self.dense1_unit_size = dense1_unit_size
        self.dense2_unit_size = dense2_unit_size
        self.directed_distance = directed_distance
        self.undirected_distance = undirected_distance
        self.source_target_dense_layers = source_target_dense_layers
        self.embedding_normalization = embedding_normalization

        hyper_params = {
            'method_name': 'siamese_graph_embedding'
        }
        hyper_params.update(kwargs)
        for key in hyper_params.keys():
            self.__setattr__('_%s' % key, hyper_params[key])

    def get_method_name(self):
        return self._method_name

    def get_method_summary(self):
        return '%s_%d' % (self._method_name, self._d)

    def create_lstm_network(self):
        """ Base network to be shared (eq. to feature extraction).
        """
        input = Input(batch_shape=(None, None))  # (batch_number, sequence_length)
        x = Embedding(5, 4, input_length=None, mask_zero=True, trainable=True)(
            input)  # (batch_number, sequence_length, 5)
        # x = Masking()(input)
        print("Embedding", x) if self.verbose else None

        x = Lambda(lambda y: K.expand_dims(y, axis=2), name="lstm_lambda_1")(x)  # (batch_number, sequence_length, 1, 5)
        x = Conv2D(filters=320, kernel_size=(self.conv1_kernel_size, 1), activation='relu',
                   data_format="channels_last", name="lstm_conv_1")(x)  # (batch_number, sequence_length-5, 1, 192)
        x = Lambda(lambda y: K.squeeze(y, axis=2), name="lstm_lambda_2")(x)  # (batch_number, sequence_length-5, 192)
        print("conv2D", x) if self.verbose else None
        if self.conv1_batch_norm:
            x = BatchNormalization(center=True, scale=True, name="conv1_batch_norm")(x)
        x = MaxPooling1D(pool_size=self.max1_pool_size, padding="same")(x)
        x = Dropout(0.2)(x)

        if self.conv2_kernel_size is not None and self.conv2_kernel_size != 0:
            x = Convolution1D(filters=192, kernel_size=self.conv2_kernel_size, activation='relu', name="lstm_conv_2")(x)
            print("conv1d_2", x) if self.verbose else None
            if self.conv2_batch_norm:
                x = BatchNormalization(center=True, scale=True, name="conv2_batch_norm")(x)
            x = MaxPooling1D(pool_size=self.max2_pool_size, padding="same")(x)
            print("max pooling_2", x) if self.verbose else None
            x = Dropout(0.2)(x)

        x = Bidirectional(CuDNNLSTM(self.lstm_unit_size, return_sequences=False, return_state=False))(x)  # (batch_number, 320+320)
        print("brnn", x) if self.verbose else None
        x = Dropout(0.5)(x)

        if self.dense1_unit_size is not None and self.dense1_unit_size != 0:
            x = Dense(self.dense1_unit_size, activation='relu', name="dense_1")(x)
            x = Dropout(0.2)(x)

        if self.source_target_dense_layers:
            source = Dense(int(self._d / 2), activation='linear', name="dense_source")(x)
            target = Dense(int(self._d / 2), activation='linear', name="dense_target")(x)
            if self.embedding_normalization:
                source = Lambda(lambda x: K.l2_normalize(x, axis=-1))(source)
                target = Lambda(lambda x: K.l2_normalize(x, axis=-1))(target)
            print("source", source) if self.verbose else None
            print("target", target) if self.verbose else None

            x = Concatenate(axis=-1, name="embedding_output")(
                [source, target])  # Embedding space (batch_number, embedding_dim)
        else:
            x = Dense(self._d, activation='linear', name="embedding_output")(x)
            if self.embedding_normalization:
                x = Lambda(lambda x: K.l2_normalize(x, axis=-1), name="embedding_output_normalized")(x)

        print("embedding", x) if self.verbose else None
        return Model(input, x, name="lstm_network")

    def create_alpha_network(self):
        encoded_i = Input(batch_shape=(None, self._d))
        encoded_j = Input(batch_shape=(None, self._d))
        is_directed = Input(batch_shape=(None, 1), dtype=tf.int8)

        abs_diff_directed = Lambda(lambda tup: K.abs(tup[0][:, 0:int(self._d/2)] - tup[1][:, int(self._d/2):self._d]),
                                   output_shape=(None, int(self._d/2)), name="Lambda_abs_diff_directed")([encoded_i, encoded_j])
        print("abs_diff_directed:", abs_diff_directed) if self.verbose else None
        abs_diff_undirected = Lambda(get_abs_diff, output_shape=(None, self._d), name="Lambda_abs_diff_directed")([encoded_i, encoded_j])
        print("abs_diff_undirected:", abs_diff_undirected) if self.verbose else None

        alpha_directed = Dense(1, activation='sigmoid',
                               # kernel_regularizer=keras.regularizers.l1(l=0.01),
                               trainable=True, name="alpha_directed")(abs_diff_directed)
        alpha_undirected = Dense(1, activation='sigmoid',
                                 # kernel_regularizer=keras.regularizers.l1(l=0.01),
                                 trainable=True, name="alpha_undirected")(abs_diff_undirected)
        print("alpha_directed:", alpha_directed) if self.verbose else None
        print("alpha_undirected:", alpha_undirected) if self.verbose else None

        output = Lambda(switch, output_shape=(None, ), name="alpha_lambda_output")([is_directed, alpha_directed, alpha_undirected])
        print("output", output) if self.verbose else None
        return Model(inputs=[encoded_i, encoded_j, is_directed], outputs=output, name="alpha_network")

    def st_euclidean_distance(self, inputs):
        emb_i, emb_j, is_directed = inputs
        sum_directed = K.sum(K.square(emb_i[:, 0:int(self._d/2)] - emb_j[:, int(self._d/2):self._d]), axis=1,
                             keepdims=True)
        sum_undirected = K.sum(K.square(emb_i - emb_j), axis=1, keepdims=True)
        sum_switch = K.switch(is_directed, sum_directed, sum_undirected)
        return K.sqrt(K.maximum(sum_switch, K.epsilon()))

    def st_min_euclidean_distance(self, inputs):
        emb_i, emb_j, is_directed = inputs
        source_i = emb_i[:, 0:int(self._d / 2)]
        target_i = emb_i[:, int(self._d / 2):self._d]
        source_j = emb_j[:, 0:int(self._d / 2)]
        target_j = emb_j[:, int(self._d / 2):self._d]

        sum_directed = K.sum(K.square(source_i - target_j), axis=1, keepdims=True)
        sum_undirected = K.sum(K.minimum(K.square(source_i - source_j), K.square(target_i - target_j)), axis=1,
                               keepdims=True)
        sum_switch = K.switch(is_directed, sum_directed, sum_undirected)
        return K.sqrt(K.maximum(sum_switch, K.epsilon()))

    def st_emb_probability(self, inputs):
        emb_i, emb_j, is_directed = inputs
        dot_directed = Lambda(lambda x: K.sum(x[0][:, 0:int(self._d/2)] * x[1][:, int(self._d/2):self._d],
                                              axis=-1, keepdims=True))([emb_i, emb_j])
        dot_undirected = Dot(axes=1, normalize=False)([emb_i, emb_j])
        return K.switch(is_directed, K.sigmoid(dot_directed), K.sigmoid(dot_undirected))

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
            input_seq_i = Input(batch_shape=(self.batch_size, None), name="input_seq_i")
            input_seq_j = Input(batch_shape=(self.batch_size, None), name="input_seq_j")
            is_directed = Input(batch_shape=(self.batch_size, 1), dtype=tf.int8, name="is_directed")

            # build create_lstm_network to use in each siamese 'leg'
            self.lstm_network = self.create_lstm_network()

            # encode each of the two inputs into a vector with the conv_lstm_network
            encoded_i = self.lstm_network(input_seq_i)
            print(encoded_i) if self.verbose else None
            encoded_j = self.lstm_network(input_seq_j)
            print(encoded_j) if self.verbose else None

            output = Lambda(
                self.st_euclidean_distance if self.undirected_distance == "euclidean" else self.st_min_euclidean_distance,
                name="lambda_output")([encoded_i, encoded_j, is_directed])
            # self.alpha_network = self.create_alpha_network()
            # output = self.alpha_network([encoded_i, encoded_j, is_directed])

            self.siamese_net = Model(inputs=[input_seq_i, input_seq_j, is_directed], outputs=output)

        # Multi-gpu parallelization
        if multi_gpu:
            self.siamese_net = multi_gpu_model(self.siamese_net, gpus=4, cpu_merge=True, cpu_relocation=False)

        # Compile & train
        self.siamese_net.compile(loss=contrastive_loss,  # binary_crossentropy, cross_entropy, contrastive_loss
                                 optimizer=RMSprop(lr=self.lr),
                                 metrics=[precision_d, recall_d] if not hasattr(self, "alpha_network") else [
                                     "accuracy", precision, recall],
                                 )
        print("Network total weights:", self.siamese_net.count_params()) if self.verbose else None

    def learn_embedding(self, network: HeterogeneousNetwork, network_val=None, multi_gpu=True,
                        n_steps=500, validation_steps=None, tensorboard=True, histogram_freq=0, embeddings=False,
                        early_stopping=2,
                        edge_f=None, is_weighted=False, no_python=False, rebuild_model=False, seed=0, **kwargs):
        generator_train = self.get_training_data_generator(network, n_steps, seed)

        if network_val is not None:
            self.generator_val = DataGenerator(network=network_val, weighted=self.weighted,
                                               maxlen=self.max_length, padding='post', truncating="post",
                                               tokenizer=generator_train.tokenizer,
                                               negative_sampling_ratio=2.0,
                                               batch_size=self.batch_size, shuffle=True, seed=seed, verbose=self.verbose) \
                if not hasattr(self, "generator_val") else self.generator_val
        else:
            self.generator_val = None

        assert generator_train.tokenizer.word_index == self.generator_val.tokenizer.word_index
        if not hasattr(self, "siamese_net") or rebuild_model: self.build_keras_model(multi_gpu)

        try:
            self.hist = self.siamese_net.fit_generator(generator_train, epochs=self.epochs,
                                                       validation_data=self.generator_val,
                                                       validation_steps=validation_steps,
                                                       callbacks=self.get_callbacks(early_stopping, tensorboard,
                                                                                    histogram_freq, embeddings),
                                                       use_multiprocessing=True, workers=16, **kwargs)
        except KeyboardInterrupt:
            print("Stop training")
        finally:
            self.save_network_weights()

    def get_training_data_generator(self, network, n_steps=500, seed=0):
        if self.subsample:
            self.generator_train = SampledDataGenerator(network=network, weighted=self.weighted,
                                                        compression_func=self.compression_func, n_steps=n_steps,
                                                        maxlen=self.max_length, padding='post',
                                                        truncating=self.truncating,
                                                        negative_sampling_ratio=self.negative_sampling_ratio,
                                                        directed_proba=self.directed_proba,
                                                        batch_size=self.batch_size, shuffle=True, seed=seed,
                                                        verbose=self.verbose) \
                if not hasattr(self, "generator_train") else self.generator_train
        else:
            self.generator_train = DataGenerator(network=network, weighted=self.weighted,
                                                 maxlen=self.max_length, padding='post', truncating=self.truncating,
                                                 negative_sampling_ratio=self.negative_sampling_ratio,
                                                 batch_size=self.batch_size, shuffle=True, seed=seed,
                                                 verbose=self.verbose) \
                if not hasattr(self, "generator_train") else self.generator_train
        self.node_list = self.generator_train.node_list
        self.word_index = self.generator_train.tokenizer.word_index
        return self.generator_train

    def build_tensorboard(self, histogram_freq, embeddings, write_grads):
        if not hasattr(self, "log_dir"):
            self.log_dir = "logs/{}_{}".format(type(self).__name__[0:20], time.strftime('%m-%d_%H:%M%p').strip(" "))
            print("log_dir:", self.log_dir)

        if embeddings:
            x_test, y_test = self.generator_val.load_data(return_node_name=True)
            if not os.path.exists(self.log_dir): os.makedirs(self.log_dir)
            with open(os.path.join(self.log_dir, "metadata.tsv"), 'w') as f:
                np.savetxt(f, y_test, fmt="%s")
                f.close()

        self.tensorboard = TensorBoard(
            log_dir=self.log_dir,
            histogram_freq=histogram_freq,
            write_grads=write_grads, write_graph=False, write_images=False,
            batch_size=self.batch_size,
            update_freq="epoch",
            embeddings_freq=1 if embeddings else 0,
            embeddings_metadata=os.path.join(self.log_dir, "metadata.tsv") if embeddings else None,
            embeddings_data=x_test if embeddings else None,
            embeddings_layer_names=["embedding_output"] if embeddings else None,
        )
        # Add params text to tensorboard

    def get_callbacks(self, early_stopping=0, tensorboard=True, histogram_freq=0, embeddings=True, write_grads=True):
        callbacks = []
        if tensorboard:
            if not hasattr(self, "tensorboard"):
                self.build_tensorboard(histogram_freq=histogram_freq, embeddings=embeddings, write_grads=write_grads)
            callbacks.append(self.tensorboard)

        if early_stopping > 0:
            if not hasattr(self, "early_stopping"):
                self.early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=early_stopping, verbose=0,
                                                    mode='auto',
                                                    baseline=None, restore_best_weights=False)
            callbacks.append(self.early_stopping)

        if len(callbacks) == 0: callbacks = None
        return callbacks

    def save_network_weights(self):
        if hasattr(self, "alpha_network"):
            self.alpha_directed = self.alpha_network.get_layer(name="alpha_directed").get_weights()
            self.alpha_undirected = self.alpha_network.get_layer(name="alpha_undirected").get_weights()

    def get_reconstructed_adj(self, beta=2.0, X=None, node_l=None, node_l_b=None, edge_type="d", var_len=False):
        """
        :param X:
        :param node_l: list of node names
        :param edge_type:
        :return:
        """
        if hasattr(self, "reconstructed_adj") and edge_type=="d":
            adj = self.reconstructed_adj
        else:
            embs = self.get_embedding(variable_length=var_len)
            assert len(self.node_list) == embs.shape[0]
            adj = self._pairwise_similarity(embs, edge_type)

        if (node_l is None or node_l == self.node_list) and node_l_b is None:
            if edge_type == "d": self.reconstructed_adj = adj
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
                # adj = np.exp(-2 * adj)
                adj = self.transform_adj_beta_exp(adj, edge_types="d", sample_negative=self.negative_sampling_ratio)
            elif self.directed_distance == "l1_alpha":
                adj = pairwise_distances(X=embeddings_X,
                                         Y=embeddings_Y,
                                         metric=l1_diff_alpha, n_jobs=-2, weights=self.alpha_directed)

        elif edge_type == 'u':
            if self.undirected_distance == "euclidean_ball":
                adj = radius_neighbors_graph(embeddings, radius=self.margin, n_jobs=-2)

            elif self.undirected_distance == "euclidean":
                adj = pairwise_distances(X=embeddings, metric="euclidean", n_jobs=-2)
                adj = self.transform_adj_beta_exp(adj, edge_types=["u", "u_n"], sample_negative=False)

            elif self.undirected_distance == "l1_alpha":
                adj = pairwise_distances(X=embeddings,
                                         metric=l1_diff_alpha, n_jobs=-2, weights=self.alpha_undirected)
        else:
            raise Exception("Unsupported edge_type", edge_type)
        return adj

    def transform_adj_beta_exp(self, adj_dist, edge_types, sample_negative):
        print("beta exp func")
        adj_true = self.generator_train.network.get_adjacency_matrix(edge_types=edge_types,
                                                                     node_list=self.node_list,
                                                                     sample_negative=sample_negative)
        rows, cols = adj_true.nonzero()
        y_true = adj_true[rows, cols]
        adj_dist_squared = adj_dist - np.min(adj_dist)
        adj_dist_squared = np.power(adj_dist_squared, 2)
        dists_pred = np.clip(adj_dist_squared[rows, cols], 1e-8, 1e8)
        beta = -np.divide(np.log(y_true), dists_pred)
        print("mean", np.mean(beta, axis=1),
              "median", np.median(beta, axis=1),
              "min", np.min(beta, axis=1),
              "max", np.max(beta, axis=1))
        beta_mean = np.median(beta, axis=1)
        print("beta_mean", beta_mean)
        adj_pred = np.exp(-np.multiply(beta_mean, adj_dist_squared))
        return adj_pred

    def save_embeddings(self, filename, logdir=True, variable_length=True, recompute=True, minlen=None):
        embs = self.get_embedding(variable_length=variable_length, recompute=recompute, minlen=minlen)
        assert len(self.node_list) == embs.shape[0]
        if logdir and hasattr(self, "log_dir"):
            file_path = os.path.join(self.log_dir, filename)
        else:
            file_path = filename
        fout = open(file_path, 'w')
        fout.write("{} {}\n".format(len(self.node_list), self._d))
        for i in range(len(self.node_list)):
            fout.write("{} {}\n".format(self.node_list[i],
                                        ' '.join([str(x) for x in embs[i]])))
        fout.close()
        print("Saved at", file_path)

    def get_embedding(self, variable_length=False, recompute=False, node_list=None, minlen=None):
        if (not hasattr(self, "_X") or recompute):
            self.process_embeddings(variable_length, batch_size=self.batch_size, minlen=minlen)

        if node_list is not None:
            idx = [self.node_list.index(node) for node in node_list if node in self.node_list]
            return self._X[idx, :]
        else:
            return self._X

    def process_embeddings(self, variable_length, batch_size=256, minlen=100):
        seqs = self.generator_train.get_sequence_data(self.node_list, variable_length=variable_length, minlen=minlen)

        if variable_length:
            embs = [self.lstm_network.predict(seq, batch_size=1) for seq in seqs]
        else:
            embs = self.lstm_network.predict(seqs, batch_size=batch_size)

        embs = np.array(embs)
        embs = embs.reshape(embs.shape[0], embs.shape[-1])
        self._X = embs

    def load_weights(self, siamese_weights, alpha_weights, generator):
        self.generator_train = generator
        self.node_list = self.generator_train.node_list
        self.build_keras_model(multi_gpu=False)
        self.lstm_network.load_weights(siamese_weights, by_name=True)
        self.alpha_network.load_weights(alpha_weights, by_name=True)
        self.save_network_weights()
        print(self.siamese_net.summary())

    def save_model(self, filename, model="lstm", logdir=True):
        if logdir and hasattr(self, "log_dir"):
            file_path = os.path.join(self.log_dir, filename)
            params_path = os.path.join(self.log_dir, "params.txt")
        else:
            file_path = filename

        if model == "lstm":
            self.lstm_network.save(file_path)
            self.write_params(params_path)
            print("Saved lstm_network model at", file_path)
        elif model == "siamese":
            self.siamese_net.save_weights(file_path)
            print("Saved siamese model weights at", file_path)

        self.write_params(params_path)

    def write_params(self, file_path):
        fout = open(file_path, 'w')
        fout.write("{}\n".format(self.get_params()))
        fout.close()

    def load_model(self, lstm_model, network):
        self.generator_train = self.get_training_data_generator(network)
        self.build_keras_model(multi_gpu=False)
        self.lstm_network = load_model(lstm_model)
        print(self.siamese_net.summary())

    def predict_generator(self, generator):
        y_pred = self.siamese_net.predict_generator(generator, use_multiprocessing=True, workers=8)
        y_pred = np.exp(-2.0*y_pred)
        return y_pred

    def get_edge_weight(self, i, j, edge_type='d'):
        if not type(i) == int or type(j) == int:
            i_idx = self.node_list.index(i)
            j_idx = self.node_list.index(j)

        return self.get_reconstructed_adj(edge_type=edge_type)[i_idx, j_idx]


if __name__ == '__main__':
    import pickle

    with open('moge/data/lncRNA_miRNA_mRNA/miRNA-mRNA_network.pickle', 'rb') as input_file:
        network = pickle.load(input_file)

    ##### Run graph embedding #####
    gf = SiameseGraphEmbedding(d=128, batch_size=10000, lr=0.05, epochs=50, reg=1.0)



    gf.learn_embedding(network)
    np.save(
        "/home/jonny_admin/PycharmProjects/MultiOmicsGraphEmbedding/moge/data/lncRNA_miRNA_mRNA/miRNA-mRNA_source_target_embeddings_128.npy",
        gf.get_embedding())