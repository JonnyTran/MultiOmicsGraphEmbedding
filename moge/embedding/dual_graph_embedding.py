import tensorflow as tf
import networkx as nx
import numpy as np
from scipy.sparse import triu
from moge.embedding.static_graph_embedding import StaticGraphEmbedding, ImportedGraphEmbedding
from moge.network.heterogeneous_network import HeterogeneousNetwork


class SourceTargetGraphEmbedding(StaticGraphEmbedding, ImportedGraphEmbedding):
    def __init__(self, d=50, lr=0.001, epochs=10, batch_size=100000, Ed_Eu_ratio=0.2, **kwargs):
        super().__init__(d)

        self._d = d
        # self.reg = reg
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

    def learn_embedding(self, network:HeterogeneousNetwork, edge_f=None,
                        is_weighted=False, no_python=False, seed=0):
        self.n_nodes = len(network.all_nodes)
        self.all_nodes = network.all_nodes

        adj_undirected = network.get_node_similarity_adjacency()
        adj_directed = network.get_regulatory_edges_adjacency()

        Ed_rows, Ed_cols = adj_directed.nonzero()  # getting the list of non-zero edges from the Sparse Numpy matrix
        Ed_count = len(Ed_rows)
        Eu_rows, Eu_cols = triu(adj_undirected, k=1).nonzero()  # only get non-zero edges from upper triangle of the adjacency matrix
        Eu_count = len(Eu_rows)

        print("Directed edges training size:", Ed_count)
        print("Undirected edges training size:", Eu_count)

        tf.reset_default_graph()
        with tf.name_scope('inputs'):
            E_ij = tf.placeholder(tf.float32, shape=(1,), name="E_ij")
            is_directed = tf.placeholder(tf.bool, name="is_directed")
            i = tf.Variable(int, name="i", trainable=False)
            j = tf.Variable(int, name="j", trainable=False)


        emb_s = tf.Variable(initial_value=tf.random_uniform([self.n_nodes, self._d], -1, 1),
                                 validate_shape=True, dtype=tf.float32,
                                 name="emb_s", trainable=True)

        emb_t = tf.Variable(initial_value=tf.random_uniform([self.n_nodes, self._d], -1, 1),
                                 validate_shape=True, dtype=tf.float32,
                                 name="emb_s", trainable=True)

        emb_c = tf.concat([emb_s, emb_t], axis=1, name="emb_concat")

        # 1st order (directed proximity)
        p_1 = tf.sigmoid(tf.matmul(tf.slice(emb_s, [i, 0], [1, emb_s.get_shape()[1]]),
                                   tf.slice(emb_t, [j, 0], [1, emb_s.get_shape()[1]]),
                                   transpose_b=True, name="p_1_inner_prod"), name="p_1")

        loss_f1 = tf.reduce_sum(-tf.multiply(E_ij, tf.log(p_1)), name="loss_f1")

        # 2nd order proximity
        p_2_exps = tf.matmul(tf.slice(emb_c, [i, 0], [1, emb_c.get_shape()[1]], name="p_2_exps_i"),
                            emb_c,
                            transpose_b=True) # dim (1, n_nodes)
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
        optimizer = tf.train.GradientDescentOptimizer(self.lr)\
            .minimize(loss, var_list=[emb_s, emb_t])

        with tf.Session() as session:
            session.as_default()
            session.run(init_op)

            if self.batch_size == None or self.batch_size == -1:
                self.batch_size = Ed_count + Eu_count

            self.iterations = int(self.epochs * (Ed_count + Eu_count) / self.batch_size)

            if self.Ed_Eu_ratio != None:
                Ed_batch_size = min(int(self.batch_size * self.Ed_Eu_ratio), Ed_count)
                Eu_batch_size = min(int(self.batch_size * (1-self.Ed_Eu_ratio)), Eu_count)
            else:
                Ed_batch_size = int(self.batch_size * Ed_count/(Ed_count + Eu_count))
                Eu_batch_size = int(self.batch_size * Eu_count/(Ed_count + Eu_count))

            print("Training", self.iterations, "iterations, with directed_edges_batch_size", Ed_batch_size, "and undirected_edges_batch_size", Eu_batch_size)

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

                    print("iteration:", step, "f1_loss", iteration_loss_f1/Ed_batch_size,
                          "f2_loss", iteration_loss_f2/Eu_batch_size)
            except KeyboardInterrupt:
                pass

            finally:
                # Save embedding vectors
                self.embedding_s = session.run([emb_s])[0].copy()
                self.embedding_t = session.run([emb_t])[0].copy()

                self._X = np.concatenate([self.embedding_s, self.embedding_t], axis=1)

                session.close()

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
        if edge_type == "d":
            estimated_adj = np.divide(1, 1 + np.power(np.e, -np.matmul(self.embedding_s, self.embedding_t.T)))
        elif edge_type == 'u':
            estimated_adj = self.softmax(np.matmul(self._X, self._X.T))
            np.fill_diagonal(estimated_adj, 0)
        else:
            raise Exception("Have not implemented directed and undirected combined adjacency matrix")

        return estimated_adj

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        fout.write("{} {}\n".format(self.n_nodes, self._d*2))
        for i in range(self.n_nodes):
            fout.write("{} {}\n".format(self.all_nodes[i],
                                        ' '.join([str(x) for x in self.get_embedding()[i]])))
        fout.close()

    def softmax(self, X):
        exps = np.exp(X)
        return exps/np.sum(exps, axis=0)

    def get_embedding(self):
        return self._X

    def get_edge_weight(self, i, j, edge_type='d'):
        if edge_type == 'd':
            return np.divide(1, 1 + np.power(np.e, -np.matmul(self.embedding_s[i], self.embedding_t[j].T)))
        elif edge_type == 'u':
            return self.softmax(np.matmul(self._X[i], self._X.T))[j]


class DualGraphEmbedding(StaticGraphEmbedding):
    def __init__(self, d=50, reg=1.0, lr=0.001, iterations=100, batch_size=1, **kwargs):
        super().__init__(d)

        self._d = d
        self.reg = reg
        self.lr = lr
        self.iterations = iterations
        self.batch_size = batch_size

        hyper_params = {
            'method_name': 'dual_graph_embedding'
        }
        hyper_params.update(kwargs)
        for key in hyper_params.keys():
            self.__setattr__('_%s' % key, hyper_params[key])

    def get_method_name(self):
        return self._method_name

    def get_method_summary(self):
        return '%s_%d' % (self._method_name, self._d)

    def learn_embedding(self, graph, edge_f=None,
                        is_weighted=False, no_python=False):
        self.n_nodes = graph.number_of_nodes()
        Y = nx.adjacency_matrix(graph)

        tf.reset_default_graph()
        with tf.name_scope('inputs'):
            y_ij = tf.placeholder(tf.float32, shape=(1,), name="y_ij")
            i = tf.Variable(int, name="i", trainable=False)
            j = tf.Variable(int, name="j", trainable=False)


        emb_s = tf.Variable(initial_value=tf.random_uniform([self.n_nodes, self._d], -1, 1),
                                 validate_shape=True, dtype=tf.float32,
                                 name="emb_s", trainable=True)

        emb_t = tf.Variable(initial_value=tf.random_uniform([self.n_nodes, self._d], -1, 1),
                                 validate_shape=True, dtype=tf.float32,
                                 name="emb_s", trainable=True)

        # print(tf.slice(emb_s, [i, 0], [1, emb_s.get_shape()[1]], name="emb_s_i"))

        p_cross = tf.sigmoid(tf.matmul(tf.slice(emb_s, [i, 0], [1, emb_s.get_shape()[1]], name="emb_s_i"),
                                       tf.slice(emb_t, [j, 0], [1, emb_s.get_shape()[1]], name="emb_t_j"),
                                       transpose_b=True, name="p_cross_inner_prod"),
                             name="p_cross")

        loss_f1 = tf.reduce_sum(-tf.multiply(y_ij, tf.log(p_cross), name="loss_f1"))

        # Add the loss value as a scalar to summary.
        tf.summary.scalar('loss', loss_f1)
        merged = tf.summary.merge_all()

        # Initialize variables
        init_op = tf.global_variables_initializer()

        # SGD Optimizer
        optimizer = tf.train.GradientDescentOptimizer(self.lr)\
            .minimize(loss_f1, var_list=[emb_s, emb_t])

        with tf.Session() as session:
            session.as_default()
            session.run(init_op)
            for step in range(self.iterations):

                rows, cols = Y.nonzero() # getting the list of non-zero edges from the Sparse Numpy matrix
                iteration_loss = 0.0
                for x, y in zip(rows, cols):
                    feed_dict = {y_ij: [Y[x, y], ],
                                 i: x,
                                 j: y}

                    _, summary, loss_val = session.run(
                        [optimizer, merged, loss_f1],
                        feed_dict=feed_dict)
                    iteration_loss += loss_val

                if step % 50 == 0:
                    print("iteration:",step, "loss", iteration_loss)

            self.embedding_s = session.run([emb_s])[0].copy()
            self.embedding_t = session.run([emb_t])[0].copy()

            session.close()

    def get_reconstructed_adj(self, X=None, node_l=None):
        return np.divide(1, 1 + np.power(np.e, -np.matmul(self.embedding_s, self.embedding_t.T)))

    def get_embedding(self):
        return np.concatenate([self.embedding_s, self.embedding_t], axis=1)
        # return self.embedding_s, self.embedding_t

    def get_edge_weight(self, i, j):
        return np.divide(1, 1 + np.power(np.e, -np.matmul(self.embedding_s[i], self.embedding_t[j].T)))



if __name__ == '__main__':
    import pickle

    with open('moge/data/lncRNA_miRNA_mRNA/miRNA-mRNA_network.pickle', 'rb') as input_file:
        network = pickle.load(input_file)

    ##### Run graph embedding #####
    gf = SourceTargetGraphEmbedding(d=64, reg=1.0, lr=0.05, epochs=50, batch_size=10000)

    gf.learn_embedding(network)
    np.save("/home/jonny_admin/PycharmProjects/MultiOmicsGraphEmbedding/moge/data/lncRNA_miRNA_mRNA/miRNA-mRNA_source_target_embeddings_128.npy",
            gf.get_embedding())
