import tensorflow as tf
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse


class GraphFactorization():

    def __init__(self, G, d=100, reg=1.0, lr=1.0):
        self.n_nodes = G.number_of_nodes()
        self.d = d
        self.reg = reg
        self.lr = lr

    def train(self, Y, iterations=100, batch_size=1):
        with tf.name_scope('inputs'):
            y_ij = tf.placeholder(tf.float32, shape=(1, ), name="y_ij")
            i = tf.Variable(int, name="i", trainable=False)
            j = tf.Variable(int, name="j", trainable=False)

        # y_ij = tf.Variable(name="y_ij")
        #
        # Y = tf.get_variable("Y", shape=(self.n_nodes, self.n_nodes),
        #                     initializer=tf.zeros_initializer())

        lr = tf.constant(self.lr)

        z_emb = tf.Variable(initial_value=tf.random_uniform([self.n_nodes, self.d], -1, 1),
                            validate_shape=True, dtype=tf.float32,
                            name="z_emb", trainable=True)

        z_ij_inner = tf.matmul(tf.transpose(tf.slice(z_emb, [i, 0], [self.d, 1], name="slice1"), name="slice1_T"),
                                                            tf.slice(z_emb, [j, 0], [self.d,1 ], name="slice2"),
                               name="z_ij_inner")

        regu_term = lr/2.0 * tf.square(tf.reduce_mean(z_emb[i]), name="regu_term")

        # Loss Function: 1/2 * sum_ij (Y_ij - <Z_i, Z_j>)^2 + lr/2 * sum_i |Z_i|^2
        loss = 1.0/2.0 * tf.add(tf.square(tf.reduce_mean(y_ij - z_ij_inner)), regu_term, name="loss")


        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            print('Loss(x,y) = %.3f' % session.run(z_ij_inner, {i:0, j:1, y_ij:[1,]}))

        # Add the loss value as a scalar to summary.
        tf.summary.scalar('loss', loss)
        merged = tf.summary.merge_all()

        # Initialize variables
        init_op = tf.global_variables_initializer()

        # SGD Optimizer
        self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(loss, var_list=[z_emb])

        with tf.Session() as session:
            session.run(init_op)
            average_loss = 0
            count = 0
            for step in range(iterations):
                print(step, count)
                count = 0

                rows, cols = Y.nonzero()
                for x,y in zip(rows,cols):
                    count+=1

                    feed_dict = {y_ij: [Y[x, y],],
                                 i: x,
                                 j: y}

                    _, summary, loss_val = session.run(
                        [self.optimizer, merged, loss],
                        feed_dict=feed_dict)

                    average_loss += loss_val

        print(average_loss)

if __name__ == '__main__':

    ppi = pd.read_table("/home/jonny_admin/PycharmProjects/nuclei-segmentation/notebooks/gem/data/BINARY_PROTEIN_PROTEIN_INTERACTIONS.txt", header=None)
    # ppi.filter([0, 3])

    G = nx.from_pandas_dataframe(ppi, source=0, target=3, create_using=nx.DiGraph())
    # nx.relabel.convert_node_labels_to_integers(G)

    gf = GraphFactorization(G, d=200, reg=1.0, lr=1.0)
    Y = nx.adjacency_matrix(G)

    gf.train(Y=Y)

