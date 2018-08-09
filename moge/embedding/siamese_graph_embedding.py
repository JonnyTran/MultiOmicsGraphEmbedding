import networkx as nx
import numpy as np

import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, LSTM
from keras.optimizers import RMSprop
from keras import backend as K


from scipy.sparse import triu
from moge.embedding.static_graph_embedding import StaticGraphEmbedding, ImportedGraphEmbedding
from moge.network.heterogeneous_network import HeterogeneousNetwork


class SiameseGraphEmbedding(StaticGraphEmbedding):
    def __init__(self, d, node_features_size, lr=0.001, epochs=10, batch_size=100000, Ed_Eu_ratio=0.2, **kwargs):
        super().__init__(d)

        self._d = d
        self.node_features_size = node_features_size
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.Ed_Eu_ratio = Ed_Eu_ratio

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
        x = LSTM(128, input_shape=input_shape, return_sequences=False)(input)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(128, activation='relu')(x)
        return Model(input, x)

    def euclidean_distance(self, vects):
        x, y = vects
        return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

    def source_target_emebedding_distance(self, vects):
        emb_i, emb_j, is_directed = vects
        dot_directed = Dot(axes=1)([emb_i[:, 0:int(_d / 2)], emb_j[:, int(_d / 2):_d]])
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
        return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

    def learn_embedding(self, network: HeterogeneousNetwork, edge_f=None, get_training_data=False,
                        is_weighted=False, no_python=False, seed=0):

        self.n_nodes = len(network.node_list)
        self.all_nodes = network.node_list

        adj_undirected = network.get_adjacency_matrix(edge_type="u", get_training_data=get_training_data)
        adj_directed = network.get_adjacency_matrix(edge_type='d', get_training_data=get_training_data)

        Ed_rows, Ed_cols = adj_directed.nonzero()  # getting the list of non-zero edges from the Sparse Numpy matrix
        Ed_count = len(Ed_rows)
        Eu_rows, Eu_cols = triu(adj_undirected, k=1).nonzero()  # only get non-zero edges from upper triangle of the adjacency matrix
        Eu_count = len(Eu_rows)

        print("Directed edges training size:", Ed_count)
        print("Undirected edges training size:", Eu_count)

        tf.reset_default_graph()

        with tf.name_scope('inputs'):
            E_ij = Input(batch_shape=(1,), name="E_ij")
            #     input_i = tf.placeholder(tf.float32, shape=input_shape, name="input_i")
            #     input_j = tf.placeholder(tf.float32, shape=input_shape, name="input_j")
            input_seq_i = Input(batch_shape=(1, None, 4), name="input_i")
            input_seq_j = Input(batch_shape=(1, None, 4), name="input_j")
            # is_directed = tf.placeholder(tf.bool, name="is_directed")
            is_directed = Input(batch_shape=(1,), dtype=tf.bool, name="is_directed")
            i = tf.Variable(int, name="i", trainable=False)
            j = tf.Variable(int, name="j", trainable=False)

        # build create_base_network to use in each siamese 'leg'
        lstm_network = self.create_base_network(input_shape=(None, 4))

        print("lstm_network", lstm_network)

        # encode each of the two inputs into a vector with the convnet
        encoded_i = lstm_network(input_seq_i)
        encoded_j = lstm_network(input_seq_j)
        print("encoded_i", encoded_i, "\nencoded_j", encoded_j)

        distance = Lambda(self.source_target_emebedding_distance)([encoded_i, encoded_j, is_directed])
        print("distance", distance)

        siamese_net = Model(inputs=[input_seq_i, input_seq_j, is_directed], outputs=distance)

        siamese_net.compile(loss=self.contrastive_loss,
                            optimizer=RMSprop(lr=0.01),
                            metrics=[self.accuracy])

        with tf.Session() as session:
            session.as_default()
            K.set_session(session)

            if self.batch_size == None or self.batch_size == -1:
                self.batch_size = Ed_count + Eu_count

            self.iterations = int(self.epochs * (Ed_count + Eu_count) / self.batch_size)

            if self.Ed_Eu_ratio != None:
                Ed_batch_size = min(int(self.batch_size * self.Ed_Eu_ratio), Ed_count)
                Eu_batch_size = min(int(self.batch_size * (1 - self.Ed_Eu_ratio)), Eu_count)
            else:
                Ed_batch_size = int(self.batch_size * Ed_count / (Ed_count + Eu_count))
                Eu_batch_size = int(self.batch_size * Eu_count / (Ed_count + Eu_count))

            print("Training", self.iterations, "iterations, with directed_edges_batch_size", Ed_batch_size,
                  "and undirected_edges_batch_size", Eu_batch_size)

            np.random.seed(seed)
            try:
                for step in range(self.iterations):
                    # Run all directed edges
                    iteration_loss_f1 = 0.0
                    for k in np.random.permutation(Ed_count)[0:Ed_batch_size]:
                        feed_dict = {E_ij: [adj_directed[Ed_rows[k], Ed_cols[k]], ],
                                     is_directed: True,
                                     i: Ed_rows[k],
                                     j: Ed_cols[k]}

                        _, summary, loss_val = session.run([optimizer, merged, loss],
                                                           feed_dict=feed_dict)
                        iteration_loss_f1 += loss_val

                    # Run all undirected edges
                    iteration_loss_f2 = 0.0
                    for k in np.random.permutation(Eu_count)[0:Eu_batch_size]:
                        feed_dict = {E_ij: [adj_undirected[Eu_rows[k], Eu_cols[k]], ],
                                     is_directed: False,
                                     i: Eu_rows[k],
                                     j: Eu_cols[k]}

                        _, summary, loss_val = session.run([optimizer, merged, loss],
                                                           feed_dict=feed_dict)
                        iteration_loss_f2 += loss_val

                    print("iteration:", step, "f1_loss", iteration_loss_f1 / Ed_batch_size,
                          "f2_loss", iteration_loss_f2 / Eu_batch_size)
            except KeyboardInterrupt:
                pass

            finally:
                # Save embedding vectors
                self.embedding_s = session.run([emb_s])[0].copy()
                self.embedding_t = session.run([emb_t])[0].copy()

                self._X = np.concatenate([self.embedding_s, self.embedding_t], axis=1)

                session.close()

    def get_training_data(self):
        return pass

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
        pass

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


class DataGenerator():

    def __init__(self) -> None:
        super().__init__()


if __name__ == '__main__':
    import pickle

    with open('moge/data/lncRNA_miRNA_mRNA/miRNA-mRNA_network.pickle', 'rb') as input_file:
        network = pickle.load(input_file)

    ##### Run graph embedding #####
    gf = SiameseGraphEmbedding(d=64, reg=1.0, lr=0.05, epochs=50, batch_size=10000)

    gf.learn_embedding(network)
    np.save(
        "/home/jonny_admin/PycharmProjects/MultiOmicsGraphEmbedding/moge/data/lncRNA_miRNA_mRNA/miRNA-mRNA_source_target_embeddings_128.npy",
        gf.get_embedding())
