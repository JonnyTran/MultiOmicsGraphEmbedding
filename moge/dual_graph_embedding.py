import tensorflow as tf
import networkx as nx


class DualGraphEmbedding():
    def __init__(self, G, d=50, reg=1.0, lr=0.001):
        self.n_nodes = G.number_of_nodes()
        self.d = d
        self.reg = reg
        self.lr = lr

    def train(self, Y, iterations=100, batch_size=1):
        with tf.name_scope('inputs'):
            y_ij = tf.placeholder(tf.float32, shape=(1,), name="y_ij")
            i = tf.Variable(int, name="i", trainable=False)
            j = tf.Variable(int, name="j", trainable=False)

        self.emb_s = tf.Variable(initial_value=tf.random_uniform([self.n_nodes, self.d], -1, 1),
                                 validate_shape=True, dtype=tf.float32,
                                 name="emb_s", trainable=True)

        self.emb_t = tf.Variable(initial_value=tf.random_uniform([self.n_nodes, self.d], -1, 1),
                                 validate_shape=True, dtype=tf.float32,
                                 name="emb_s", trainable=True)

        print(tf.slice(self.emb_s, [i, 0], [1, self.emb_s.get_shape()[1]], name="emb_s_i"))

        p_cross = tf.sigmoid(tf.matmul(tf.slice(self.emb_s, [i, 0], [1, self.emb_s.get_shape()[1]], name="emb_s_i"),
                                       tf.slice(self.emb_t, [j, 0], [1, self.emb_s.get_shape()[1]], name="emb_t_j"),
                                       transpose_b=True, name="p_cross_inner_prod"),
                             name="p_cross")

        self.loss_f1 = tf.reduce_sum(-tf.multiply(y_ij, tf.log(p_cross), name="loss_f1"))

        # Add the loss value as a scalar to summary.
        tf.summary.scalar('loss', self.loss_f1)
        merged = tf.summary.merge_all()

        # Initialize variables
        init_op = tf.global_variables_initializer()

        # SGD Optimizer
        self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss_f1,
                                                                             var_list=[self.emb_s, self.emb_t])

        with tf.Session() as session:
            session.as_default()
            session.run(init_op)
            for step in range(iterations):
                print("iteration", step)
                rows, cols = Y.nonzero()
                count = 0.0
                interation_loss = 0.0
                for x, y in zip(rows, cols):
                    count += 1
                    feed_dict = {y_ij: [Y[x, y], ],
                                 i: x,
                                 j: y}

                    _, summary, loss_val = session.run(
                        [self.optimizer, merged, self.loss_f1],
                        feed_dict=feed_dict)
                    interation_loss += loss_val

                print(interation_loss / count)

            print(self.emb_s.read_value())


if __name__ == '__main__':
    G = nx.read_edgelist("/Users/jonny/Desktop/PycharmProjects/MultiOmicsGraphEmbedding/data/karate.edgelist",
                         create_using=nx.DiGraph())

    # G = nx.from_pandas_edgelist(ppi, source=0, target=3, create_using=nx.DiGraph())
    # nx.relabel.convert_node_labels_to_integers(G)

    gf = DualGraphEmbedding(G, d=5, reg=1.0, lr=0.05)
    Y = nx.adjacency_matrix(G)

    gf.train(Y=Y, iterations=10)
