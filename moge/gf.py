import tensorflow as tf
import networkx as nx
import numpy as np
import pandas as pd


class GraphFactorization():

    def __init__(self, G, d=100, reg=1.0, lr=1.0):
        self.n_nodes = G.number_of_nodes()
        self.d = d
        self.reg = reg
        self.lr = lr

    def train(self, Y, iterations=100, batch_size=1):
        with tf.name_scope('inputs'):
            y_ij_inputs = tf.placeholder(tf.float32, shape=[batch_size])
            i = tf.placeholder(tf.int32, shape=[1])
            j = tf.placeholder(tf.int32, shape=[1])

        # y_ij = tf.Variable(name="y_ij")
        #
        # Y = tf.get_variable("Y", shape=(self.n_nodes, self.n_nodes),
        #                     initializer=tf.zeros_initializer())

        lr = tf.constant(self.lr)

        z_emb = tf.Variable(
            tf.random_uniform(
                [self.n_nodes, self.d], -1, 1),
            name="z_emb")

        z_ind = tf.range(self.d)

        # Loss Function
        loss = 1.0 / 2 * tf.square(tf.reduce_mean(y_ij_inputs - tf.matmul(tf.transpose(z_emb)[i], z_emb[j]))) + \
               lr / 2 * tf.square(tf.reduce_mean(z_emb, axis=1))

        # Add the loss value as a scalar to summary.
        tf.summary.scalar('loss', loss)

        # SGD Optimizer
        self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(loss)

        merged = tf.summary.merge_all()

        with tf.Session() as session:
            average_loss = 0
            for step in range(iterations):

                for (x, y), value in np.ndenumerate(Y):
                    feed_dict = {y_ij_inputs: value, i: x, j: y}

                    _, summary, loss_val = session.run(
                        [self.optimizer, merged, loss],
                        feed_dict=feed_dict)

                    average_loss += loss_val


if __name__ == '__main__':

    ppi = pd.read_table("/home/jonny_admin/PycharmProjects/nuclei-segmentation/notebooks/gem/data/BINARY_PROTEIN_PROTEIN_INTERACTIONS.txt", header=None)
    ppi.filter([0, 3])
    np.savetxt(r'gem/data/ppi.edgelist', ppi.filter([0, 3]).values, fmt='%s')
    G = nx.read_edgelist('gem/data/ppi.edgelist', create_using=nx.DiGraph())
    # nx.relabel.convert_node_labels_to_integers(G)

    gf = GraphFactorization(G, d=100, reg=1.0, lr=1.0)
    gf.train(Y=nx.adjacency_matrix(G))

