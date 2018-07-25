import networkx as nx
import numpy as np

import tensorflow as tf
from keras import backend as K
from keras.layers import Dense

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
            E_ij = tf.placeholder(tf.float32, shape=(1,), name="E_ij")
            N_i = tf.placeholder(tf.float32, shape=(1, self.node_features_size), name="N_i")
            N_j = tf.placeholder(tf.float32, shape=(1, self.node_features_size), name="N_j")
            is_directed = tf.placeholder(tf.bool, name="is_directed")
            i = tf.Variable(int, name="i", trainable=False)
            j = tf.Variable(int, name="j", trainable=False)


        # Siamese network
        with tf.name_scope('siamese'):
            emb_c_i = Dense(128, activation='relu')(N_i)
            emb_c_i = Dense(128, activation='relu')(emb_c_i)

            emb_c_j = Dense(128, activation='relu')(N_j)
            emb_c_j = Dense(128, activation='relu')(emb_c_j)

        # emb_s = tf.Variable(initial_value=tf.random_uniform([self.n_nodes, self._d], -1, 1),
        #                     validate_shape=True, dtype=tf.float32,
        #                     name="emb_s", trainable=True)

        # emb_t = tf.Variable(initial_value=tf.random_uniform([self.n_nodes, self._d], -1, 1),
        #                     validate_shape=True, dtype=tf.float32,
        #                     name="emb_s", trainable=True)

        emb_c = tf.concat([emb_s, emb_t], axis=1, name="emb_concat")

        # 1st order (directed proximity)
        p_1 = tf.sigmoid(tf.matmul(tf.slice(emb_s, [i, 0], [1, emb_s.get_shape()[1]]),
                                   tf.slice(emb_t, [j, 0], [1, emb_s.get_shape()[1]]),
                                   transpose_b=True, name="p_1_inner_prod"), name="p_1")

        loss_f1 = tf.reduce_sum(-tf.multiply(E_ij, tf.log(p_1)), name="loss_f1")

        # 2nd order proximity
        p_2_exps = tf.matmul(tf.slice(emb_c, [i, 0], [1, emb_c.get_shape()[1]], name="p_2_exps_i"),
                             emb_c,
                             transpose_b=True)  # dim (1, n_nodes)
        p_2 = tf.slice(tf.nn.softmax(p_2_exps - tf.reduce_max(p_2_exps, axis=1),
                                     axis=1, name="p_2_softmax"),
                       [0, j], [1, 1], "p_2_i_j")

        loss_f2 = tf.reduce_sum(-tf.multiply(E_ij, tf.log(p_2)), name="loss_f2")

        loss = tf.cond(is_directed, true_fn=lambda: loss_f1, false_fn=lambda: loss_f2)

        # Add the loss value as a scalar to summary.
        tf.summary.scalar('loss', loss)
        merged = tf.summary.merge_all()

        # Initialize variables
        init_op = tf.global_variables_initializer()

        # SGD Optimizer
        optimizer = tf.train.GradientDescentOptimizer(self.lr) \
            .minimize(loss, var_list=[emb_s, emb_t])

        with tf.Session() as session:
            session.as_default()
            K.set_session(session)
            session.run(init_op)

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
