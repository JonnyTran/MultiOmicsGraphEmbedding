import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Dropout, Input, Lambda, LSTM, Bidirectional
from keras.layers import Dot, MaxPooling1D, Convolution1D
from keras.models import Model
from keras.optimizers import RMSprop

from moge.embedding.static_graph_embedding import StaticGraphEmbedding
from moge.network.data_generator import DataGenerator
from moge.network.heterogeneous_network import HeterogeneousNetwork


class SiameseGraphEmbedding(StaticGraphEmbedding):
    def __init__(self, d, input_shape, batch_size=100, lr=0.001, epochs=10,
                 max_length=700, Ed_Eu_ratio=0.2, **kwargs):
        super().__init__(d)

        self._d = d
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.Ed_Eu_ratio = Ed_Eu_ratio
        self.max_length = max_length

        hyper_params = {
            'method_name': 'source_target_graph_embedding'
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
        print("conv1d", x)
        x = MaxPooling1D(pool_size=13, strides=13)(x)  # Similar to DanQ Model
        print("max pooling", x)
        x = Dropout(0.2)(x)
        x = Bidirectional(LSTM(320, return_sequences=False, return_state=False))(x)
        print("brnn", x)
        x = Dropout(0.5)(x)
        #     x = GlobalMaxPooling1D()(x)
        #     print("GAP pooling", x)

        x = Dense(75 * 640, activation='relu')(x)
        x = Dense(925, activation='sigmoid')(x)
        x = Dense(self._d, activation='linear')(x)  # Embedding space
        return Model(input, x)

    def euclidean_distance(self, inputs):
        x, y = inputs
        return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

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

    def learn_embedding(self, network: HeterogeneousNetwork, n_epochs=10,
                        edge_f=None, get_training_data=False,
                        is_weighted=False, no_python=False, seed=0):

        self.generator = DataGenerator(network=network, get_training_data=get_training_data,
                                  maxlen=self.max_length, padding='post', truncating="post",
                                  batch_size=self.batch_size, dim=self.input_shape, shuffle=True)

        # Inputs
        E_ij = Input(batch_shape=(self.batch_size, 1), name="E_ij")
        input_seq_i = Input(batch_shape=(self.batch_size, *self.input_shape), name="input_seq_i")
        input_seq_j = Input(batch_shape=(self.batch_size, *self.input_shape), name="input_seq_j")
        is_directed = Input(batch_shape=(self.batch_size, 1), dtype=tf.bool, name="is_directed")

        # build create_base_network to use in each siamese 'leg'
        self.lstm_network = self.create_base_network(input_shape=(self.max_length, 6))

        # encode each of the two inputs into a vector with the convnet
        encoded_i = self.lstm_network(input_seq_i)
        encoded_j = self.lstm_network(input_seq_j)

        distance = Lambda(self.st_emb_probability)([encoded_i, encoded_j, is_directed])

        self.siamese_net = Model(inputs=[input_seq_i, input_seq_j, is_directed], outputs=distance)
        self.siamese_net.compile(loss=self.contrastive_loss,
                            optimizer=RMSprop(lr=self.lr),
                            metrics=[self.accuracy])

        print("Network total weights:", self.siamese_net.count_params())

        self.siamese_net.fit_generator(self.generator, use_multiprocessing=True, workers=9, epochs=n_epochs)


    def get_reconstructed_adj(self, X=None, node_l=None, edge_type="d"):
        """
        For inter-modality, we calculate the directed first-order proximity, for intra-modality, we calculate the
        second-order proximity.
        The combined will be the adjacency matrix.

        :param X:
        :param node_l:
        :param edge_type:
        :return:
        """
        pass #TODO

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        fout.write("{} {}\n".format(self.n_nodes, self._d * 2))
        for i in range(self.n_nodes):
            fout.write("{} {}\n".format(self.all_nodes[i],
                                        ' '.join([str(x) for x in self.get_embedding()[i]])))
        fout.close()

    def softmax(self, X):
        exps = np.exp(X)
        return exps / np.sum(exps, axis=0)

    def get_embedding(self):
        pass

    def get_edge_weight(self, i, j, edge_type='d'):
        pass




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
