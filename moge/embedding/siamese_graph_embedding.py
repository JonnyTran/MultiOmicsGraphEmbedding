import numpy as np
from sklearn.metrics import pairwise_distances
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Dropout, Input, Lambda, LSTM, Bidirectional
from keras.layers import Dot, MaxPooling1D, Convolution1D
from keras.models import Model
from keras.optimizers import RMSprop
from keras.utils import multi_gpu_model
from keras.models import load_model

from moge.embedding.static_graph_embedding import StaticGraphEmbedding, ImportedGraphEmbedding
from moge.network.data_generator import DataGenerator
from moge.network.heterogeneous_network import HeterogeneousNetwork

from sklearn.preprocessing import MinMaxScaler

class SiameseGraphEmbedding(ImportedGraphEmbedding):
    def __init__(self, d=512, input_shape=(None, 6), batch_size=1024, lr=0.001, epochs=10,
                 max_length=700, Ed_Eu_ratio=0.2, seed=0, verbose=False, **kwargs):
        super().__init__(d)

        self._d = d
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.Ed_Eu_ratio = Ed_Eu_ratio
        self.max_length = max_length
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

    def create_base_network(self, input_shape):
        """ Base network to be shared (eq. to feature extraction).
        """
        input = Input(shape=input_shape)
        #     x = Flatten()(input)
        x = Convolution1D(filters=320, kernel_size=26, input_shape=input_shape, activation='relu')(input)
        print("conv1d", x) if self.verbose else None
        x = MaxPooling1D(pool_size=13, strides=13)(x)  # Similar to DanQ Model
        print("max pooling", x) if self.verbose else None
        x = Dropout(0.2)(x)
        x = Bidirectional(LSTM(320, return_sequences=False, return_state=False))(x)  # TODO Return states?
        print("brnn", x) if self.verbose else None
        x = Dropout(0.5)(x)
        #     x = GlobalMaxPooling1D()(x)
        #     print("GAP pooling", x)

        x = Dense(75 * 640, activation='relu')(x)
        x = Dense(925, activation='relu')(x)
        x = Dense(self._d, activation='linear')(x)  # Embedding space
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

    def contrastive_loss(self, y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        margin = 1
        return K.mean(y_true * K.square(y_pred) +
                      (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

    def accuracy(self, y_true, y_pred):
        '''Compute classification accuracy with a fixed threshold on distances.
        '''
        return K.mean(K.equal(K.cast(y_true > 0.5, y_true.dtype), K.cast(y_pred > 0.8, y_true.dtype)))

    def learn_embedding(self, network: HeterogeneousNetwork, network_val=None, multi_gpu=False,
                        edge_f=None, is_weighted=False, no_python=False, seed=0):

        self.generator = DataGenerator(network=network,
                                       maxlen=self.max_length, padding='post', truncating="post",
                                       batch_size=self.batch_size, dim=self.input_shape, shuffle=True)
        self.node_list = self.generator.node_list

        if network_val:
            generator_val = DataGenerator(network=network_val,
                                          maxlen=self.max_length, padding='post', truncating="post",
                                          batch_size=self.batch_size, dim=self.input_shape, shuffle=True)
        else:
            generator_val = None

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
            self.lstm_network = self.create_base_network(input_shape=self.input_shape)

            # encode each of the two inputs into a vector with the convnet
            encoded_i = self.lstm_network(input_seq_i)
            encoded_j = self.lstm_network(input_seq_j)

            distance = Lambda(self.st_euclidean_distance)([encoded_i, encoded_j, is_directed])

            self.siamese_net = Model(inputs=[input_seq_i, input_seq_j, is_directed], outputs=distance)

        # Multi-gpu parallelization
        if multi_gpu:
            self.siamese_net = multi_gpu_model(self.siamese_net, gpus=4, cpu_merge=True, cpu_relocation=False)

        # Compile & train
        self.siamese_net.compile(loss=self.contrastive_loss,
                            optimizer=RMSprop(lr=self.lr),
                            metrics=[self.accuracy])

        print("Network total weights:", self.siamese_net.count_params()) if self.verbose else None

        self.history = self.siamese_net.fit_generator(self.generator, epochs=self.n_epochs,
                                                      validation_data=generator_val,
                                                      use_multiprocessing=True, workers=8)


    def get_reconstructed_adj(self, X=None, node_l=None, edge_type="d"):
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
        if edge_type == 'd':
            adj = pairwise_distances(X=embs[:, 0:int(self._d / 2)],
                                     Y=embs[:, int(self._d / 2):self._d],
                                     metric="euclidean", n_jobs=8)
        else:
            adj = pairwise_distances(X=embs, metric="euclidean", n_jobs=8)

        adj = MinMaxScaler(feature_range=(0, 1), copy=True).fit_transform(-adj)

        return adj

    def save_embeddings(self, filepath):
        fout = open(filepath, 'w')
        fout.write("{} {}\n".format(len(self.node_list), self._d * 2))
        for i in range(len(self.node_list)):
            fout.write("{} {}\n".format(len(self.node_list)[i],
                                        ' '.join([str(x) for x in self.get_embedding()[i]])))
        fout.close()

    def load_model(self, filepath):
        self.lstm_network = load_model(filepath)
        print(self.lstm_network.summary())


    def softmax(self, X):
        exps = np.exp(X)
        return exps / np.sum(exps, axis=0)

    def get_embedding(self, variable_length=False, recompute=False):
        if not hasattr(self, "_X") or recompute:
            seqs = self.generator.get_sequence_data(range(len(self.generator.node_list)),
                                                    variable_length=variable_length)
            if variable_length:
                embs = [self.lstm_network.predict(seq, batch_size=1) for seq in seqs]
            else:
                embs = self.lstm_network.predict(seqs, batch_size=256)

            embs = np.array(embs)
            embs = embs.reshape(embs.shape[0], embs.shape[-1])
            self._X = embs
        else:
            return self._X

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
