from abc import ABCMeta
from multiprocessing import cpu_count, Pool

import numpy as np
import pandas as pd
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from moge.evaluation.clustering import _get_top_enrichr_term, chunkIt
from moge.evaluation.utils import get_scalefree_fit_score, largest_indices


class BaseGraphEmbedding:
    __metaclass__ = ABCMeta

    def __init__(self, d):
        '''Initialize the Embedding class

        Args:
            d: dimension of model
        '''
        pass

    def get_method_name(self):
        ''' Returns the name for the model method

        Return:
            The name of model
        '''
        return ''

    def get_method_summary(self):
        ''' Returns the summary for the model include method name and paramater setting

        Return:
            A summary string of the method
        '''
        return ''

    def learn_embedding(self, graph):
        '''Learning the graph model from the adjcency matrix.

        Args:
            graph: the graph to embed in networkx DiGraph format
        '''
        pass

    def get_embeddings(self):
        ''' Returns the learnt model

        Return:
            A numpy array of size #nodes * d
        '''
        pass

    def get_node_list(self):
        return self.node_list

    def get_edge_weight(self, i, j):
        '''Compute the weight for edge between node i and node j

        Args:
            i, j: two node id in the graph for model
        Returns:
            A single number represent the weight of edge between node i and node j

        '''
        pass

    def is_trained(self):
        if hasattr(self, "_X"):
            return True
        else:
            return False

    def get_reconstructed_adj(self, edge_type=None):
        '''Compute the adjacency matrix from the learned model

        Returns:
            A numpy array of size #nodes * #nodes containing the reconstructed adjacency matrix.
        '''
        pass


class ImportedGraphEmbedding(BaseGraphEmbedding):
    __metaclass__ = ABCMeta

    def __init__(self, d, method_name="ImportedGraphEmbedding"):
        '''Initialize the Embedding class

        Args:
            d: dimension of model
        '''
        self.embedding_d = d
        self._method_name = method_name

    def get_method_name(self):
        ''' Returns the name for the model method

        Return:
            The name of model
        '''
        return self._method_name

    def get_method_summary(self):
        ''' Returns the summary for the model include method name and paramater setting

        Return:
            A summary string of the method
        '''
        return self._method_name + str(self.embedding_d)

    def learn_embedding(self, graph):
        '''Learning the graph model from the adjcency matrix.

        Args:
            graph: the graph to embed in networkx DiGraph format
        '''
        raise NotImplementedError()

    def get_embeddings(self, node_list=None):
        ''' Returns the learnt model

        Return:
            A numpy array of size #nodes * d
        '''
        if node_list is None:
            return self._X
        elif set(node_list) <= set(self.node_list):
            idx = [self.node_list.index(node) for node in node_list]
            return self._X[idx, :]
        else:
            raise Exception("node_list contains a node not included in trained embeddings")

    def get_edge_weight(self, i, j):
        '''Compute the weight for edge between node i and node j

        Args:
            i, j: two node id in the graph for model
        Returns:
            A single number represent the weight of edge between node i and node j

        '''
        raise NotImplementedError()

    def save_embeddings(self, file_path):
        embs = self.get_embeddings()
        assert len(self.node_list) == embs.shape[0]
        fout = open(file_path, 'w')
        fout.write("{} {}\n".format(len(self.node_list), self.embedding_d))
        for i in range(len(self.node_list)):
            fout.write("{} {}\n".format(self.node_list[i],
                                        ' '.join([str(x) for x in embs[i]])))
        fout.close()
        print("Saved at", file_path)

    def import_embedding(self, file, node_list):
        self.imported = True
        with open(file, "r") as fin:
            node_num, size = [int(x) for x in fin.readline().strip().split()]
            vectors = {}
            self.node_list = []

            # Read model file
            while 1:
                l = fin.readline()
                if l == '':
                    break
                vec = l.strip().split(' ')
                assert len(vec) == size + 1
                vectors[vec[0]] = [float(x) for x in vec[1:]]
            fin.close()
            assert len(vectors) == node_num

            if self.get_method_name() == "rna2rna":
                self.embedding_d = size
                self.embedding_s = []
                self.embedding_t = []

                for node in node_list:
                    if node in vectors.keys():
                        self.embedding_s.append(vectors[node][0: int(self.embedding_d / 2)])
                        self.embedding_t.append(vectors[node][int(self.embedding_d / 2): int(self.embedding_d)])
                        self.node_list.append(node)

                self.embedding_s = np.array(self.embedding_s)
                self.embedding_t = np.array(self.embedding_t)
                self._X = np.concatenate([self.embedding_s, self.embedding_t], axis=1)

            else:
                self.embedding_d = size
                self._X = []
                for node in node_list:
                    if node in vectors.keys():
                        self._X.append(vectors[node])
                        self.node_list.append(node)
                self._X = np.array(self._X)

        print(self.get_method_name(), "imported", self._X.shape)

    def get_reconstructed_adj(self, edge_type=None, node_l=None, node_l_b=None, interpolate=False):
        '''Compute the adjacency matrix from the learned model

        Returns:
            A numpy array of size #nodes * #nodes containing the reconstructed adjacency matrix.
        '''
        if hasattr(self, "reconstructed_adj"):
            reconstructed_adj = self.reconstructed_adj

        elif self._method_name == "LINE":
            # reconstructed_adj = np.divide(1, 1 + np.exp(-np.matmul(self._X, self._X.T)))
            reconstructed_adj = self.softmax(np.matmul(self._X, self._X.T))

        elif self._method_name == "node2vec":
            reconstructed_adj = self.softmax(np.matmul(self._X, self._X.T))

        elif self._method_name == "BioVec":
            reconstructed_adj = self.softmax(np.matmul(self._X, self._X.T))  # TODO Double check paper
            # reconstructed_adj = reconstructed_adj.T  # Transpose matrix since there's a bug

        elif self._method_name == "rna2rna":
            reconstructed_adj = pairwise_distances(X=self._X[:, 0:int(self.embedding_d / 2)],
                                                   Y=self._X[:, int(self.embedding_d / 2):self.embedding_d],
                                                   metric="euclidean", n_jobs=-2)
            reconstructed_adj = reconstructed_adj.T
            reconstructed_adj = self.transform_adj_adaptive_threshold(reconstructed_adj, self.network)
            # reconstructed_adj = self.transform_adj_beta_exp(reconstructed_adj, network_train=self.network,
            #                                                 edge_types="d", sample_negative=1.0)
            # reconstructed_adj = np.exp(-2.0 * reconstructed_adj)
        elif self._method_name == "HOPE":
            reconstructed_adj = np.matmul(self._X[:, 0:int(self.embedding_d / 2)],
                                          self._X[:, int(self.embedding_d / 2):self.embedding_d].T)
            interpolate = True
        elif self._method_name == "SDNE":
            reconstructed_adj = pairwise_distances(X=self._X, Y=self._X, metric="euclidean", n_jobs=-2)
            reconstructed_adj = np.exp(-1.0 * reconstructed_adj)

        else:
            raise Exception("Method" + self.get_method_name() + "not supported")

        if interpolate:
            reconstructed_adj = np.interp(reconstructed_adj, (reconstructed_adj.min(), reconstructed_adj.max()), (0, 1))

        if node_l is None or node_l == self.node_list and node_l_b is None:
            self.reconstructed_adj = reconstructed_adj
            return reconstructed_adj
        elif set(node_l) < set(self.node_list) or node_l_b is not None:
            return self._select_adj_indices(reconstructed_adj, node_l, node_l_b)
        elif not (set(node_l) < set(self.node_list)):
            raise Exception("A node in node_l is not in self.node_list.")

    def _select_adj_indices(self, adj, node_list_A, node_list_B=None):
        if node_list_B is None:
            idx = [self.node_list.index(node) for node in node_list_A]
            return adj[idx, :][:, idx]
        else:
            idx_A = [self.node_list.index(node) for node in node_list_A]
            idx_B = [self.node_list.index(node) for node in node_list_B]
            return adj[idx_A, :][:, idx_B]

    @DeprecationWarning
    def transform_adj_adaptive_threshold(self, adj_pred, network_train, margin=0.01, edge_types="d"):
        print("adaptive threshold")
        adj_true = network_train.get_adjacency_matrix(edge_types=edge_types, node_list=self.node_list)
        self.distance_threshold_nodes = self.get_adaptive_threshold(adj_pred, adj_true, margin)
        print("distance_threshold", self.distance_threshold_nodes)
        adj_pred = np.where(adj_pred < self.distance_threshold_nodes, 1, 0)
        return adj_pred

    @DeprecationWarning
    def get_adaptive_threshold(self, adj_pred, adj_true, margin):
        distance_threshold = np.zeros((len(self.node_list),))
        for nonzero_node_id in np.unique(adj_true.nonzero()[0]):
            _, nonzero_node_cols = adj_true[nonzero_node_id].nonzero()
            positive_distances = adj_pred[nonzero_node_id, nonzero_node_cols]
            distance_threshold[nonzero_node_id] = np.min(positive_distances)
        median_threshold = np.min(distance_threshold[distance_threshold > 0])
        distance_threshold[distance_threshold == 0] = median_threshold
        return distance_threshold + margin

    @DeprecationWarning
    def transform_adj_beta_exp(self, adj_dist, network_train, edge_types, sample_negative):
        print("beta exp func")
        adj_true = network_train.get_adjacency_matrix(edge_types=edge_types, node_list=self.node_list,
                                                      sample_negative=sample_negative)
        rows, cols = adj_true.nonzero()
        y_true = adj_true[rows, cols]
        adj_dist_squared = adj_dist - np.min(adj_dist)
        adj_dist_squared = np.power(adj_dist_squared, 2)
        dists_pred = np.clip(adj_dist_squared[rows, cols], 1e-8, 1e8)
        beta = -np.divide(np.log(y_true), dists_pred)
        beta_mean = np.median(beta, axis=1)
        print("beta mean", np.mean(beta, axis=1), "beta median", np.median(beta, axis=1))
        adj_pred = np.exp(-np.multiply(beta_mean, adj_dist_squared))
        return adj_pred

    def softmax(self, X):
        exps = np.exp(X)
        return exps/np.sum(exps, axis=0)

    def get_top_k_predicted_edges(self, edge_type, top_k, node_list=None, node_list_B=None, training_network=None,
                                  databases=None):
        nodes = self.node_list
        if node_list is not None and node_list_B is not None:
            nodes = [n for n in nodes if n in node_list or n in node_list_B]
            nodes_A = [n for n in self.node_list if n in node_list]
            nodes_B = [n for n in self.node_list if n in node_list_B]
        elif node_list is not None:
            nodes = [n for n in nodes if n in node_list]

        if node_list_B is not None:
            estimated_adj = self.get_reconstructed_adj(edge_type=edge_type, node_l=nodes_A,
                                                       node_l_b=nodes_B)  # (node_list_A, node_list_B)
        else:
            estimated_adj = self.get_reconstructed_adj(edge_type=edge_type, node_l=nodes)  # (nodes, nodes)
        np.fill_diagonal(estimated_adj, 0)

        if training_network is not None:
            training_adj = training_network.get_adjacency_matrix(edge_types=[edge_type], node_list=nodes,
                                                                 databases=databases)
            if node_list_B is not None:
                idx_A = [nodes.index(node) for node in nodes_A]
                idx_B = [nodes.index(node) for node in nodes_B]
                training_adj = training_adj[idx_A, :][:, idx_B]
            assert estimated_adj.shape == training_adj.shape, "estimated_adj {} != training_adj {}".format(
                estimated_adj.shape, training_adj.shape)
            rows, cols = training_adj.nonzero()
            estimated_adj[rows, cols] = 0

        top_k_indices = largest_indices(estimated_adj, top_k, smallest=False)

        if node_list_B is not None:
            top_k_pred_edges = [(nodes_A[x[0]], nodes_B[x[1]], estimated_adj[x[0], x[1]]) for x in zip(*top_k_indices)]
        else:
            top_k_pred_edges = [(nodes[x[0]], nodes[x[1]], estimated_adj[x[0], x[1]]) for x in zip(*top_k_indices)]

        return top_k_pred_edges

    def get_bipartite_adj(self, node_list_A, node_list_B):
        nodes_A = [n for n in self.node_list if n in node_list_A]
        nodes_B = [n for n in self.node_list if n in node_list_B]
        nodes = list(set(nodes_A) | set(nodes_B))

        estimated_adj = self.get_reconstructed_adj(node_l=nodes)
        assert len(nodes) == estimated_adj.shape[0]
        nodes_A_idx = [nodes.index(node) for node in nodes_A if node in nodes]
        nodes_B_idx = [nodes.index(node) for node in nodes_B if node in nodes]
        bipartite_adj = estimated_adj[nodes_A_idx, :][:, nodes_B_idx]
        return bipartite_adj

    def get_scalefree_fit_score(self, node_list_A, node_list_B=None, k_power=1, plot=False):
        if node_list_B is not None:
            adj = self.get_bipartite_adj(node_list_A, node_list_B)
        else:
            adj = self.get_reconstructed_adj(node_l=node_list_A)

        degrees_A = np.sum(adj, axis=1)

        return get_scalefree_fit_score(degrees_A, k_power, plot)

    def predict(self, X):
        """
        Bulk predict whether an edge exists between a pair of nodes, provided as a collection in X.
        :param X: [n_pairs, 2], where each index along axis 0 is a tuple containing a pair of nodes.
        :return y_pred: [n_pairs]
        """
        estimated_adj = self.get_reconstructed_adj()
        node_set = set(self.node_list)
        # node_set_in_X = set(node for pair in X for node in pair)

        estimated_adj[0, 0] = 0.0
        X_u_inx = [self.node_list.index(u) if (u in node_set and v in node_set) else 0 for u, v in X]
        X_v_inx = [self.node_list.index(v) if (u in node_set and v in node_set) else 0 for u, v in X]
        y_pred = estimated_adj[X_u_inx, X_v_inx]
        y_pred = np.array(y_pred, dtype=np.float).reshape((-1, 1))
        assert y_pred.shape[0] == X.shape[0]
        return y_pred

    def process_tsne_node_pos(self):
        embs = self.get_embeddings()
        embs_pca = PCA(n_components=2).fit_transform(embs)
        self.node_pos = TSNE(init=embs_pca, n_jobs=8).fit_transform(embs)

    def get_tsne_node_pos(self):
        if hasattr(self, "node_pos"):
            return self.node_pos
        else:
            self.process_tsne_node_pos()
            return self.node_pos

    def predict_cluster(self, n_clusters=8, node_list=None, n_jobs=-2, save_kmeans=False):
        embs = self.get_embeddings()
        kmeans = KMeans(n_clusters, n_jobs=n_jobs)
        y_pred = kmeans.fit_predict(embs)
        if save_kmeans: self.kmeans = kmeans

        if node_list is not None and set(node_list) <= set(self.node_list):
            idx = [self.node_list.index(node) for node in node_list]
            y_pred = np.array(y_pred)[idx]
        else:
            assert node_list == self.node_list

        return y_pred

    def get_cluster_members(self, cluster):
        return [self.node_list[node_index] for node_index in np.where(self.kmeans.labels_ == cluster)[0]]

    def get_cluster_neighbors(self, node):
        return self.get_cluster_members(self.kmeans.labels_[self.node_list.index(node)])

    def get_cluster_assignment(self, node_list=None):
        y_pred = self.kmeans.labels_
        assert y_pred.shape[0] == len(self.node_list)
        if node_list is not None and set(node_list) <= set(self.node_list) and node_list != self.node_list:
            idx = [self.node_list.index(node) for node in node_list]
            y_pred = y_pred[idx]

        return y_pred.tolist()

    def get_top_enrichr_term(self):
        gene_sets = [self.get_cluster_members(cluster_num) for cluster_num in range(self.kmeans.n_clusters)]
        data_split = chunkIt(gene_sets, cpu_count())
        print(len(data_split), len(data_split[0]))
        pool = Pool(cpu_count())
        results = pool.map(_get_top_enrichr_term, data_split)
        df = pd.concat(results)
        df["Adjusted P-value"] = df["Adjusted P-value"].astype(float)
        pool.close()
        pool.join()
        return df.sort_values(by="Adjusted P-value")


class NeuralGraphEmbedding(ImportedGraphEmbedding):
    def __init__(self, d, method_name="ImportedGraphEmbedding"):
        super(NeuralGraphEmbedding, self).__init__(d, method_name)

    def create_encoder_network(self, **kwargs):
        raise NotImplementedError()

    def create_embedding_model(self, **kwargs):
        raise NotImplementedError()

    def create_cls_model(self, **kwargs):
        raise NotImplementedError()

    def learn_embedding(self, generator_train, generator_test, **kwargs):
        raise NotImplementedError()

    def get_callbacks(self, early_stopping=0, tensorboard=True, histogram_freq=0, embeddings=False, write_grads=False):
        raise NotImplementedError()

    def save_model(self, filename, model="lstm", logdir=True):
        raise NotImplementedError()

    def load_weights(self):
        raise NotImplementedError()
