
import time
import numpy as np
import tensorflow as tf
from keras import backend as K
import keras
from keras.callbacks import TensorBoard, EarlyStopping
from keras.layers import Conv2D, Dense, Dropout, Bidirectional, CuDNNLSTM, SpatialDropout1D, Embedding
from keras.layers import Dot, MaxPooling1D, Convolution1D
from keras.layers import Input, Lambda, Activation, Subtract, Reshape
from keras.constraints import NonNeg

from keras.models import Model
from keras.models import load_model
from keras.optimizers import RMSprop
from keras.utils import multi_gpu_model
from sklearn.metrics import pairwise_distances

from moge.embedding.static_graph_embedding import ImportedGraphEmbedding
from moge.evaluation.metrics import accuracy_d, precision_d, recall_d, auc_roc_d, precision, recall, auc_roc
from moge.network.data_generator import DataGenerator, SampledDataGenerator
from moge.network.heterogeneous_network import HeterogeneousNetwork


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1.0
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



class SiameseGraphEmbedding(ImportedGraphEmbedding):
    def __init__(self, d=128, batch_size=2048, lr=0.001, epochs=10,
                 negative_sampling_ratio=2.0,
                 max_length=1400, truncating="post", seed=0, verbose=False, **kwargs):
        super().__init__(d)

        self._d = d
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.negative_sampling_ratio = negative_sampling_ratio
        self.max_length = max_length
        self.truncating = truncating
        self.seed = seed
        self.verbose = verbose

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
        x = Embedding(5, 4, input_length=None, mask_zero=True, trainable=True)(input)  # (batch_number, sequence_length, 5)
        # x = Masking()(input)
        print("Embedding", x) if self.verbose else None

        x = Lambda(lambda y: K.expand_dims(y, axis=2), name="lstm:lambda_1")(x)  # (batch_number, sequence_length, 1, 5)
        x = Conv2D(filters=192, kernel_size=(6, 1), activation='relu', data_format="channels_last")(
            x)  # (batch_number, sequence_length-5, 1, 192)
        x = Lambda(lambda y: K.squeeze(y, axis=2), name="lstm:lambda_2")(x)  # (batch_number, sequence_length-5, 192)
        print("conv2D", x) if self.verbose else None

        x = MaxPooling1D(pool_size=6, padding="same")(x)
        print("max pooling_1", x) if self.verbose else None
        x = SpatialDropout1D(0.1)(x)

        x = Convolution1D(filters=320, kernel_size=3, activation='relu')(x)
        print("conv1d_2", x) if self.verbose else None
        x = MaxPooling1D(pool_size=3, padding="same")(x)
        print("max pooling_2", x) if self.verbose else None
        x = SpatialDropout1D(0.1)(x)

        x = Bidirectional(CuDNNLSTM(320, return_sequences=False, return_state=False))(x)  # (batch_number, 320+320)
        print("brnn", x) if self.verbose else None
        x = Dropout(0.2)(x)

        x = Dense(1024, activation='relu')(x)  # (batch_number, 1024)
        x = Dropout(0.2)(x)
        x = Dense(925, activation='relu')(x)  # (batch_number, 925)
        x = Dropout(0.2)(x)
        x = Dense(self._d, activation='linear', name="embedding_output")(x)  # Embedding space (batch_number, 128)
        print("embedding", x) if self.verbose else None
        return Model(input, x, name="lstm_network")

    def create_alpha_network(self):
        encoded_i = Input(batch_shape=(None, self._d))
        encoded_j = Input(batch_shape=(None, self._d))
        is_directed = Input(batch_shape=(None, 1), dtype=tf.int8)

        abs_diff_directed = Lambda(lambda tup: K.abs(tup[0][:, 0:int(self._d/2)] - tup[1][:, int(self._d/2):self._d]),
                                   output_shape=(None, int(self._d/2)), name="Lambda_abs_diff_directed")([encoded_i, encoded_j])
        print("abs_diff_directed:", abs_diff_directed)
        abs_diff_undirected = Lambda(get_abs_diff, output_shape=(None, self._d), name="Lambda_abs_diff_directed")([encoded_i, encoded_j])
        print("abs_diff_undirected:", abs_diff_undirected)

        alpha_directed = Dense(1, activation='sigmoid',
                               # kernel_regularizer=keras.regularizers.l1(l=0.01),
                               trainable=True, name="alpha_directed")(abs_diff_directed)
        alpha_undirected = Dense(1, activation='sigmoid',
                                 # kernel_regularizer=keras.regularizers.l1(l=0.01),
                                 trainable=True, name="alpha_undirected")(abs_diff_undirected)
        print("alpha_directed:", alpha_directed)
        print("alpha_undirected:", alpha_undirected)

        output = Lambda(switch, output_shape=(None, ), name="alpha:lambda_output")([is_directed, alpha_directed, alpha_undirected])
        print("output", output)
        return Model(inputs=[encoded_i, encoded_j, is_directed], outputs=output, name="alpha_network")


    def st_euclidean_distance(self, inputs):
        emb_i, emb_j, is_directed = inputs
        sum_directed = K.sum(K.square(emb_i[:, 0:int(self._d/2)] - emb_j[:, int(self._d/2):self._d]), axis=1,
                             keepdims=True)
        sum_undirected = K.sum(K.square(emb_i - emb_j), axis=1, keepdims=True)
        sum_switch = K.switch(is_directed, sum_directed, sum_undirected)
        return K.sqrt(K.maximum(sum_switch, K.epsilon()))


    def st_emb_probability(self, inputs):
        emb_i, emb_j, is_directed = inputs
        dot_directed = Lambda(lambda x: K.sum(x[0][:, 0:int(self._d/2)] * x[1][:, int(self._d/2):self._d],
                                              axis=-1, keepdims=True))([emb_i, emb_j])
        dot_undirected = Dot(axes=1, normalize=False)([emb_i, emb_j])
        return K.switch(is_directed, K.sigmoid(dot_directed), K.sigmoid(dot_undirected))


    def build_keras_model(self, multi_gpu):
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
            print(encoded_i)
            encoded_j = self.lstm_network(input_seq_j)
            print(encoded_j)

            output = Lambda(self.st_euclidean_distance, name="lambda_output")([encoded_i, encoded_j, is_directed])
            # self.alpha_network = self.create_alpha_network()
            # output = self.alpha_network([encoded_i, encoded_j, is_directed])

            self.siamese_net = Model(inputs=[input_seq_i, input_seq_j, is_directed], outputs=output)

        # Multi-gpu parallelization
        if multi_gpu:
            self.siamese_net = multi_gpu_model(self.siamese_net, gpus=4, cpu_merge=True, cpu_relocation=False)


        # Compile & train
        self.siamese_net.compile(loss=contrastive_loss,  # binary_crossentropy, cross_entropy, contrastive_loss
                                 optimizer=RMSprop(lr=self.lr, decay=0.01),
                                 metrics=[accuracy_d, precision_d, recall_d, auc_roc_d],
                                 # metrics=["accuracy", precision, recall],
                                 )
        print("Network total weights:", self.siamese_net.count_params()) if self.verbose else None

    def learn_embedding(self, network: HeterogeneousNetwork, network_val=None, validation_make_data=False, multi_gpu=False,
                        subsample=True, compression_func="log", directed_proba=0.8,
                        n_steps=500, validation_steps=None,
                        edge_f=None, is_weighted=False, no_python=False, seed=0):
        if subsample:
            self.generator_train = SampledDataGenerator(network=network, compression_func=compression_func, n_steps=n_steps,
                                                        maxlen=self.max_length, padding='post', truncating=self.truncating,
                                                        negative_sampling_ratio=self.negative_sampling_ratio,
                                                        directed_proba=directed_proba,
                                                        batch_size=self.batch_size, shuffle=True, seed=0) \
                if not hasattr(self, "generator_train") else self.generator_train
        else:
            self.generator_train = DataGenerator(network=network,
                                                 maxlen=self.max_length, padding='post', truncating=self.truncating,
                                                 negative_sampling_ratio=self.negative_sampling_ratio,
                                                 batch_size=self.batch_size, shuffle=True, seed=0) \
                if not hasattr(self, "generator_train") else self.generator_train
        self.node_list = self.generator_train.node_list

        if network_val is not None:
            self.generator_val = DataGenerator(network=network_val,
                                               maxlen=self.max_length, padding='post', truncating="post",
                                               negative_sampling_ratio=1.0,
                                               batch_size=self.batch_size, shuffle=True, seed=0) \
                if not hasattr(self, "generator_val") else self.generator_val
        else:
            self.generator_val = None

        if not hasattr(self, "siamese_net"): self.build_keras_model(multi_gpu)

        self.tensorboard = TensorBoard(log_dir="logs/{}".format(time.strftime('%m-%d_%l-%M%p')), histogram_freq=1,
                                       write_grads=True, write_graph=False, write_images=False,
                                        batch_size=self.batch_size,
                                       # update_freq=100000, embeddings_freq=1,
                                       # embeddings_data=self.generator_val.__getitem__(0)[0],
                                       # embeddings_layer_names=["embedding_output"],
                                       )

        try:
            self.history = self.siamese_net.fit_generator(self.generator_train, epochs=self.epochs,
                                                          validation_data=self.generator_val.make_dataset(
                                                              return_sequence_data=True, batch_size=5) if validation_make_data else self.generator_val,
                                                          validation_steps=validation_steps,
                                                          callbacks=[self.tensorboard],
                                                          use_multiprocessing=True, workers=8)
        except KeyboardInterrupt:
            print("Stop training")
        finally:
            self.save_alpha_layers()

    def save_alpha_layers(self):
        if hasattr(self, "alpha_network"):
            self.alpha_directed = self.alpha_network.get_layer(name="alpha_directed").get_weights()
            self.alpha_undirected = self.alpha_network.get_layer(name="alpha_undirected").get_weights()

    def get_reconstructed_adj(self, beta=2.0, X=None, node_l=None, edge_type="d"):
        """
        :param X:
        :param node_l: list of node names
        :param edge_type:
        :return:
        """
        if hasattr(self, "reconstructed_adj") and edge_type=="d":
            adj = self.reconstructed_adj
        else:
            embs = self.get_embedding()
            assert len(self.node_list) == embs.shape[0]

            if edge_type == 'd':
                adj = pairwise_distances(X=embs[:, 0:int(self._d / 2)],
                                         Y=embs[:, int(self._d / 2):self._d],
                                         metric="euclidean", n_jobs=-2)
                                         # metric=l1_diff_alpha, n_jobs=-2, weights=self.alpha_directed)
                adj = np.exp(-2.0 * adj)
            elif edge_type == 'u':
                adj = pairwise_distances(X=embs,
                                         metric="euclidean", n_jobs=-2)
                                         # metric=l1_diff_alpha, n_jobs=-2, weights=self.alpha_undirected)
                adj = np.exp(-2.0 * adj)
            else:
                raise Exception("Unsupported edge_type", edge_type)

        adj = adj.T # Transpose matrix since there's a bug

        if (node_l is None or node_l == self.node_list):
            if edge_type=="d": self.reconstructed_adj = adj

            return adj
        elif set(node_l) < set(self.node_list):
            idx = [self.node_list.index(node) for node in node_l]
            return adj[idx, :][:, idx]
        else:
            raise Exception("A node in node_l is not in self.node_list.")

    def save_embeddings(self, filepath, variable_length=True, recompute=True, minlen=None):
        embs = self.get_embedding(variable_length=variable_length, recompute=recompute, minlen=minlen)
        assert len(self.node_list) == embs.shape[0]
        fout = open(filepath, 'w')
        fout.write("{} {}\n".format(len(self.node_list), self._d))
        for i in range(len(self.node_list)):
            fout.write("{} {}\n".format(self.node_list[i],
                                        ' '.join([str(x) for x in embs[i]])))
        fout.close()

    def get_embedding(self, variable_length=False, recompute=False, batch_size=1, node_list=None, minlen=None):
        if (not hasattr(self, "_X") or recompute):
            self.process_embeddings(batch_size, variable_length, minlen=minlen)

        if node_list is not None:
            idx = [self.node_list.index(node) for node in node_list if node in self.node_list]
            return self._X[idx, :]
        else:
            return self._X

    def load_weights(self, siamese_weights, alpha_weights, generator):
        self.generator_train = generator
        self.node_list = self.generator_train.node_list
        self.build_keras_model(multi_gpu=False)
        self.lstm_network.load_weights(siamese_weights, by_name=True)
        self.alpha_network.load_weights(alpha_weights, by_name=True)
        self.save_alpha_layers()
        print(self.siamese_net.summary())

    def load_model(self, lstm_model, generator):
        self.generator_train = generator
        self.node_list = self.generator_train.node_list
        self.build_keras_model(multi_gpu=False)
        self.lstm_network = load_model(lstm_model)
        print(self.siamese_net.summary())

    def process_embeddings(self, batch_size, variable_length, minlen=None):
        if isinstance(self.generator_train, SampledDataGenerator):
            nodelist = self.generator_train.node_list
        else:
            nodelist = range(len(self.generator_train.node_list))

        seqs = self.generator_train.get_sequence_data(nodelist,
                                                      variable_length=variable_length, minlen=minlen)
        if variable_length:
            embs = [self.lstm_network.predict(seq, batch_size=1) for seq in seqs]
        else:
            embs = self.lstm_network.predict(seqs, batch_size=batch_size)

        embs = np.array(embs)
        embs = embs.reshape(embs.shape[0], embs.shape[-1])
        self._X = embs

    def predict_generator(self, generator):
        y_pred = self.siamese_net.predict_generator(generator, use_multiprocessing=True, workers=8)
        y_pred = np.exp(-2.0*y_pred)
        return y_pred

    def get_edge_weight(self, i, j, edge_type='d'):
        embs = self.get_embedding()
        if not type(i) is int and type(j) is int:
            i = self.node_list.index(i)
            j = self.node_list.index(j)

        if edge_type == 'd':
            return pairwise_distances(X=embs[i, 0:int(self._d / 2)],
                                      Y=embs[j, int(self._d / 2):self._d],
                                      metric=l1_diff_alpha, weights=self.alpha_directed)
        else:
            return pairwise_distances(X=embs[i], Y=embs[j], metric=l1_diff_alpha, weights=self.alpha_directed)


if __name__ == '__main__':
    import pickle

    with open('moge/data/lncRNA_miRNA_mRNA/miRNA-mRNA_network.pickle', 'rb') as input_file:
        network = pickle.load(input_file)

    ##### Run graph embedding #####
    gf = SiameseGraphEmbedding(d=128, reg=1.0, lr=0.05, epochs=50, batch_size=10000)



    gf.learn_embedding(network)
    np.save(
        "/home/jonny_admin/PycharmProjects/MultiOmicsGraphEmbedding/moge/data/lncRNA_miRNA_mRNA/miRNA-mRNA_source_target_embeddings_128.npy",
        gf.get_embedding())