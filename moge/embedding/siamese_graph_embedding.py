
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, Dense, Dropout, Bidirectional, CuDNNLSTM, SpatialDropout1D, Embedding
from keras.layers import Dot, MaxPooling1D, Convolution1D
from keras.layers import Input, Lambda
from keras.models import Model
from keras.models import load_model
from keras.optimizers import RMSprop
from keras.utils import multi_gpu_model
from sklearn.metrics import pairwise_distances

from moge.embedding.static_graph_embedding import ImportedGraphEmbedding
from moge.evaluation.metrics import accuracy, precision, recall, auc_roc
from moge.network.data_generator import DataGenerator, SampledDataGenerator
from moge.network.heterogeneous_network import HeterogeneousNetwork


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


class SiameseGraphEmbedding(ImportedGraphEmbedding):

    def __init__(self, d=128, input_shape=(None, None), batch_size=2048, lr=0.001, epochs=10,
                 negative_sampling_ratio=2.0,
                 max_length=1400, truncating="post", seed=0, verbose=False, **kwargs):
        super().__init__(d)

        self._d = d
        self.input_shape = input_shape
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

    def create_base_network(self):
        """ Base network to be shared (eq. to feature extraction).
        """
        input = Input(batch_shape=(None, None))  # (batch_number, sequence_length)
        x = Embedding(6, 5, input_length=None, mask_zero=True, trainable=True)(
            input)  # (batch_number, sequence_length, 5)
        # x = Masking()(input)
        print("Embedding", x) if self.verbose else None

        x = Lambda(lambda y: K.expand_dims(y, axis=2))(x)  # (batch_number, sequence_length, 1, 5)
        x = Conv2D(filters=192, kernel_size=(6, 1), activation='relu', data_format="channels_last")(
            x)  # (batch_number, sequence_length-5, 1, 192)
        x = Lambda(lambda y: K.squeeze(y, axis=2))(x)  # (batch_number, sequence_length-5, 192)
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
        x = Dense(self._d, activation='linear')(x)  # Embedding space (batch_number, 128)
        print("embedding", x) if self.verbose else None
        return Model(input, x)

    def euclidean_distance(self, inputs):
        x, y = inputs
        return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

    def st_euclidean_distance(self, inputs):
        emb_i, emb_j, is_directed = inputs
        sum_directed = K.sum(K.square(emb_i[:, 0:int(self._d / 2)] - emb_j[:, int(self._d / 2):self._d]), axis=1,
                             keepdims=True)
        sum_undirected = K.sum(K.square(emb_i - emb_j), axis=1, keepdims=True)
        sum_switch = K.switch(is_directed, sum_directed, sum_undirected)
        return K.sqrt(K.maximum(sum_switch, K.epsilon()))


    def st_emb_probability(self, inputs):
        emb_i, emb_j, is_directed = inputs
        dot_directed = Dot(axes=1)([emb_i[:, 0:int(self._d / 2)], emb_j[:, int(self._d / 2):self._d]])
        dot_undirected = Dot(axes=1)([emb_i, emb_j])
        return K.switch(is_directed, K.sigmoid(dot_directed), K.sigmoid(dot_undirected))

    def learn_embedding(self, network: HeterogeneousNetwork, network_val=None, multi_gpu=False,
                        subsample=True, compression_func="log",
                        n_steps=500, validation_steps=None,
                        edge_f=None, is_weighted=False, no_python=False, seed=0):
        self.subsample = subsample
        if subsample:
            self.generator_train = SampledDataGenerator(network=network, compression_func=compression_func, n_steps=n_steps,
                                                        maxlen=self.max_length, padding='post', truncating=self.truncating,
                                                        negative_sampling_ratio=self.negative_sampling_ratio,
                                                        batch_size=self.batch_size, shuffle=True, seed=0)
        else:
            self.generator_train = DataGenerator(network=network,
                                                 maxlen=self.max_length, padding='post', truncating=self.truncating,
                                                 negative_sampling_ratio=self.negative_sampling_ratio,
                                                 batch_size=self.batch_size, shuffle=True, seed=0)

        self.node_list = self.generator_train.node_list

        if network_val:
            self.generator_val = DataGenerator(network=network_val,
                                               maxlen=self.max_length, padding='post', truncating="post",
                                               negative_sampling_ratio=2.0,
                                               batch_size=self.batch_size, shuffle=True, seed=0)
        else:
            self.generator_val = None

        self.build_keras_model(multi_gpu)

        self.history = self.siamese_net.fit_generator(self.generator_train, epochs=self.epochs,
                                                      validation_data=self.generator_val,
                                                      validation_steps=validation_steps,
                                                      use_multiprocessing=True, workers=8)

    def build_keras_model(self, multi_gpu):
        if multi_gpu:
            device = "/cpu:0"
            allow_soft_placement = True
        else:
            device = "/gpu:0"
            allow_soft_placement = False
        K.clear_session()
        tf.reset_default_graph()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=allow_soft_placement))
        # Build model
        with tf.device(device):
            # Inputs
            E_ij = Input(batch_shape=(self.batch_size, 1), name="E_ij")
            input_seq_i = Input(batch_shape=(self.batch_size, *self.input_shape), name="input_seq_i")
            input_seq_j = Input(batch_shape=(self.batch_size, *self.input_shape), name="input_seq_j")
            is_directed = Input(batch_shape=(self.batch_size, 1), dtype=tf.bool, name="is_directed")

            # build create_base_network to use in each siamese 'leg'
            self.lstm_network = self.create_base_network()

            # encode each of the two inputs into a vector with the convnet
            encoded_i = self.lstm_network(input_seq_i)
            encoded_j = self.lstm_network(input_seq_j)

            distance = Lambda(self.st_euclidean_distance)([encoded_i, encoded_j, is_directed])

            self.siamese_net = Model(inputs=[input_seq_i, input_seq_j, is_directed], outputs=distance)
        # Multi-gpu parallelization
        if multi_gpu:
            self.siamese_net = multi_gpu_model(self.siamese_net, gpus=4, cpu_merge=True, cpu_relocation=False)
        my_callbacks = [EarlyStopping(monitor='auc_roc', patience=300, verbose=1, mode='max')]
        # Compile & train
        self.siamese_net.compile(loss=contrastive_loss,
                                 optimizer=RMSprop(lr=self.lr),
                                 metrics=[accuracy, precision, recall, auc_roc])
        print("Network total weights:", self.siamese_net.count_params()) if self.verbose else None

    def get_reconstructed_adj(self, beta=2.0, X=None, node_l=None, edge_type="d"):
        """
        For inter-modality, we calculate the directed first-order proximity, for intra-modality, we calculate the
        second-order proximity.
        The combined will be the adjacency matrix.

        :param X:
        :param node_l: list of node names
        :param edge_type:
        :return:
        """
        embs = self.get_embedding()
        assert len(self.node_list) == embs.shape[0]
        if node_l is not None:
            indices = [i for i in range(embs.shape[0]) if self.node_list[i] in node_l]
            embs = embs[indices, :]

        if edge_type == 'd':
            adj = pairwise_distances(X=embs[:, 0:int(self._d / 2)],
                                     Y=embs[:, int(self._d / 2):self._d],
                                     metric="euclidean", n_jobs=8)
        elif edge_type == 'u':
            adj = pairwise_distances(X=embs, metric="euclidean", n_jobs=8)

        adj = np.exp(-beta * adj)

        return adj

    def save_embeddings(self, filepath, variable_length=True, recompute=True):
        embs = self.get_embedding(variable_length=variable_length, recompute=recompute)
        assert len(self.node_list) == embs.shape[0]
        fout = open(filepath, 'w')
        fout.write("{} {}\n".format(len(self.node_list), self._d))
        for i in range(len(self.node_list)):
            fout.write("{} {}\n".format(self.node_list[i],
                                        ' '.join([str(x) for x in embs[i]])))
        fout.close()

    def load_model(self, filepath, generator):
        self.generator_train = generator
        self.node_list = self.generator_train.node_list
        # self.build_keras_model(multi_gpu=False)
        self.lstm_network = load_model(filepath)
        print(self.lstm_network.summary())

    def softmax(self, X):
        exps = np.exp(X)
        return exps / np.sum(exps, axis=0)

    def get_embedding(self, variable_length=False, recompute=False, batch_size=1, node_list=None):
        if (not hasattr(self, "_X") or recompute):
            self.process_embeddings(batch_size, variable_length)

        if node_list is not None:
            idx = [self.node_list.index(node) for node in node_list if node in self.node_list]
            return self._X[idx, :]
        else:
            return self._X

    def process_embeddings(self, batch_size, variable_length):
        if self.generator_train is SampledDataGenerator:
            nodelist = self.generator_train.node_list
        else:
            nodelist = range(len(self.generator_train.node_list))
        seqs = self.generator_train.get_sequence_data(nodelist,
                                                      variable_length=variable_length)
        if variable_length:
            embs = [self.lstm_network.predict(seq, batch_size=1) for seq in seqs]
        else:
            embs = self.lstm_network.predict(seqs, batch_size=batch_size)

        embs = np.array(embs)
        embs = embs.reshape(embs.shape[0], embs.shape[-1])
        self._X = embs

    def predict_generator(self, generator):
        y_pred = self.siamese_net.predict_generator(generator, use_multiprocessing=True, workers=8)
        y_prob_pred = np.exp(-2.0 * y_pred)
        return y_prob_pred

    def get_edge_weight(self, i, j, edge_type='d'):
        embs = self.get_embedding()

        if edge_type == 'd':
            return pairwise_distances(X=embs[i, 0:int(self._d / 2)],
                                      Y=embs[j, int(self._d / 2):self._d],
                                      metric="euclidean", n_jobs=8)
        else:
            return pairwise_distances(X=embs[i], Y=embs[j], metric="euclidean", n_jobs=8)




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