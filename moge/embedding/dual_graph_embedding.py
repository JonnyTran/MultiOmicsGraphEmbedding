import tensorflow as tf
import networkx as nx
import numpy as np
from moge.embedding.static_graph_embedding import StaticGraphEmbedding
from moge.network.heterogeneous_network import HeterogeneousNetwork
from TCGAMultiOmics.multiomics import MultiOmicsData

class SourceTargetGraphEmbedding(StaticGraphEmbedding):
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

    def learn_embedding(self, network:HeterogeneousNetwork, edge_f=None,
                        is_weighted=False, no_python=False):
        self.n_nodes = len(network.all_nodes)
        E_u = network.get_node_similarity_adjacency()
        E_d = network.get_regulatory_edges_adjacency()



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


        p_1 = tf.sigmoid(tf.matmul(tf.slice(emb_s, [i, 0], [1, emb_s.get_shape()[1]], name="emb_s_i"),
                                       tf.slice(emb_t, [j, 0], [1, emb_s.get_shape()[1]], name="emb_t_j"),
                                       transpose_b=True, name="p_1_inner_prod"), name="p_1")

        loss_f1 = tf.reduce_sum(-tf.multiply(E_ij, tf.log(p_1), name="loss_f1"))

        p_2 = tf.sigmoid(tf.matmul(tf.slice(tf.concat(emb_s, emb_t, 0), [i, 0], [1, emb_s.get_shape()[1]*2], name="emb_s_i"),
                                       tf.slice(tf.concat(emb_s, emb_t, 0), [j, 0], [1, emb_s.get_shape()[1]*2], name="emb_t_j"),
                                       transpose_b=True, name="p_2_inner_prod"), name="p_2")

        loss_f2 = tf.reduce_sum(-tf.multiply(E_ij, tf.log(p_2), name="loss_f2"))

        loss = tf.cond(is_directed, true_fn=loss_f1, false_fn=loss_f2)

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

            E_d_rows, E_d_cols = E_d.nonzero() # getting the list of non-zero edges from the Sparse Numpy matrix
            E_u_rows, E_u_cols = E_u.nonzero()  # getting the list of non-zero edges from the Sparse Numpy matrix

            print("Directed edges", len(E_d_rows))
            print("Undirected edges", len(E_u_rows))

            for step in range(self.iterations):

                iteration_loss = 0.0
                for x, y in zip(E_d_rows, E_d_cols):
                    feed_dict = {E_ij: [E_d[x, y], ],
                                 is_directed: True,
                                 i: x,
                                 j: y}

                    _, summary, loss_val = session.run(
                        [optimizer, merged, loss],
                        feed_dict=feed_dict)
                    iteration_loss += loss_val
                print("iteration:",step, "f1_loss", iteration_loss)


                iteration_loss = 0.0
                for x, y in zip(E_u_rows, E_u_cols):
                    feed_dict = {E_ij: [E_u[x, y], ],
                                 is_directed: False,
                                 i: x,
                                 j: y}

                    _, summary, loss_val = session.run(
                        [optimizer, merged, loss],
                        feed_dict=feed_dict)
                    iteration_loss += loss_val
                print("iteration:", step, "f1_loss", iteration_loss)

            # Save embedding vectors
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
    folder_path = "/home/jonny_admin/PycharmProjects/Bioinformatics_ExternalData/tcga-assembler/LUAD"
    external_data_path = "/home/jonny_admin/PycharmProjects/Bioinformatics_ExternalData/"
    luad_data = MultiOmicsData(cancer_type="LUAD", tcga_data_path=folder_path, external_data_path=external_data_path,
                               modalities=[
                                   "GE",
                                   "MIR",
                                   # "LNC",
                                   # "CNV",
                                   # "SNP",
                                   #                                        "PRO",
                                   # "DNA"
                               ])

    luad_data.GE.drop_genes(set(luad_data.GE.get_genes_list()) & set(luad_data.LNC.get_genes_list()))

    network = HeterogeneousNetwork(modalities=["MIR", "GE"], multi_omics_data=luad_data)
    # Adds mRNA-mRNA and miRNA-miRNA node similarity
    network.add_edges_from_nodes_similarity(modality="GE", similarity_threshold=0.99,
                                            features=["locus_type", "gene_family_id"])
    network.add_edges_from_nodes_similarity(modality="MIR", similarity_threshold=0.5)

    # Adds miRNA-target interaction network
    network.add_edges_from_edgelist(edgelist=luad_data.MIR.get_miRNA_target_interaction_edgelist(),
                                    modalities=["MIR", "GE"])
    # Adds Gene Regulatory Network edges
    network.add_edges_from_edgelist(edgelist=luad_data.GE.get_RegNet_GRN_edgelist(),
                                    modalities=["GE", "GE"])
    network.remove_isolates()

    gf = SourceTargetGraphEmbedding(d=64, reg=1.0, lr=0.05, iterations=100)

    gf.learn_embedding(network)
    np.save("/home/jonny_admin/PycharmProjects/MultiOmicsGraphEmbedding/moge/data/lncRNA_miRNA_mRNA/miRNA-mRNA_source_target_embeddings_128.npy",
            gf.get_embedding())
