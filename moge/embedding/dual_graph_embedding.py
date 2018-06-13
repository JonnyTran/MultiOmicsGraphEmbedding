import tensorflow as tf
import networkx as nx
import numpy as np
from moge.embedding.static_graph_embedding import StaticGraphEmbedding

class DualGraphEmbedding(StaticGraphEmbedding):
    def __init__(self, d=50, reg=1.0, lr=0.001, iterations=100, batch_size=1):
        super().__init__(d)

        self.d = d
        self.reg = reg
        self.lr = lr
        self.iterations = iterations
        self.batch_size = batch_size

    def learn_embedding(self, graph, edge_f=None,
                        is_weighted=False, no_python=False):
        self.n_nodes = graph.number_of_nodes()
        Y = nx.adjacency_matrix(graph)

        with tf.name_scope('inputs'):
            y_ij = tf.placeholder(tf.float32, shape=(1,), name="y_ij")
            i = tf.Variable(int, name="i", trainable=False)
            j = tf.Variable(int, name="j", trainable=False)


        emb_s = tf.Variable(initial_value=tf.random_uniform([self.n_nodes, self.d], -1, 1),
                                 validate_shape=True, dtype=tf.float32,
                                 name="emb_s", trainable=True)

        emb_t = tf.Variable(initial_value=tf.random_uniform([self.n_nodes, self.d], -1, 1),
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
    G = nx.read_edgelist("/home/jonny_admin/PycharmProjects/MultiOmicsGraphEmbedding/moge/data/karate.edgelist",
                         create_using=nx.DiGraph())

    # G = nx.from_pandas_edgelist(ppi, source=0, target=3, create_using=nx.DiGraph())
    # nx.relabel.convert_node_labels_to_integers(G)

    gf = DualGraphEmbedding(d=5, reg=1.0, lr=0.05, iterations=100)

    gf.learn_embedding(G)
    print(gf.get_embedding())
