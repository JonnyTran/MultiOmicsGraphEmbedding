from abc import ABCMeta

import numpy as np
from sklearn.metrics import pairwise_distances
from moge.evaluation.link_prediction import largest_indices


class StaticGraphEmbedding:
    __metaclass__ = ABCMeta

    def __init__(self, d):
        '''Initialize the Embedding class

        Args:
            d: dimension of embedding
        '''
        pass

    def get_method_name(self):
        ''' Returns the name for the embedding method

        Return:
            The name of embedding
        '''
        return ''

    def get_method_summary(self):
        ''' Returns the summary for the embedding include method name and paramater setting

        Return:
            A summary string of the method
        '''
        return ''

    def learn_embedding(self, graph):
        '''Learning the graph embedding from the adjcency matrix.

        Args:
            graph: the graph to embed in networkx DiGraph format
        '''
        pass

    def get_embedding(self):
        ''' Returns the learnt embedding

        Return:
            A numpy array of size #nodes * d
        '''
        pass

    def get_node_list(self):
        return self.node_list

    def get_edge_weight(self, i, j):
        '''Compute the weight for edge between node i and node j

        Args:
            i, j: two node id in the graph for embedding
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
        '''Compute the adjacency matrix from the learned embedding

        Returns:
            A numpy array of size #nodes * #nodes containing the reconstructed adjacency matrix.
        '''
        pass

class ImportedGraphEmbedding(StaticGraphEmbedding):
    __metaclass__ = ABCMeta

    def __init__(self, d, method_name="ImportedGraphEmbedding"):
        '''Initialize the Embedding class

        Args:
            d: dimension of embedding
        '''
        self._d = d
        self._method_name = method_name

    def get_method_name(self):
        ''' Returns the name for the embedding method

        Return:
            The name of embedding
        '''
        return self._method_name

    def get_method_summary(self):
        ''' Returns the summary for the embedding include method name and paramater setting

        Return:
            A summary string of the method
        '''
        return self._method_name + str(self._d)

    def learn_embedding(self, graph):
        '''Learning the graph embedding from the adjcency matrix.

        Args:
            graph: the graph to embed in networkx DiGraph format
        '''
        pass

    def get_embedding(self):
        ''' Returns the learnt embedding

        Return:
            A numpy array of size #nodes * d
        '''
        return self._X

    def get_edge_weight(self, i, j):
        '''Compute the weight for edge between node i and node j

        Args:
            i, j: two node id in the graph for embedding
        Returns:
            A single number represent the weight of edge between node i and node j

        '''
        pass

    def import_embedding(self, file, node_list):
        self.imported = True
        with open(file, "r") as fin:
            node_num, size = [int(x) for x in fin.readline().strip().split()]
            vectors = {}
            self.node_list = []

            # Read embedding file
            while 1:
                l = fin.readline()
                if l == '':
                    break
                vec = l.strip().split(' ')
                assert len(vec) == size + 1
                vectors[vec[0]] = [float(x) for x in vec[1:]]
            fin.close()
            assert len(vectors) == node_num

            if self.get_method_name() == "source_target_graph_embedding":
                self._d = size
                self.embedding_s = []
                self.embedding_t = []

                for node in node_list:
                    if node in vectors.keys():
                        self.embedding_s.append(vectors[node][0 : int(self._d/2)])
                        self.embedding_t.append(vectors[node][int(self._d/2) : int(self._d)])
                        self.node_list.append(node)

                self.embedding_s = np.array(self.embedding_s)
                self.embedding_t = np.array(self.embedding_t)
                self._X = np.concatenate([self.embedding_s, self.embedding_t], axis=1)

            else:
                self._d = size
                self._X = []
                for node in node_list:
                    if node in vectors.keys():
                        self._X.append(vectors[node])
                        self.node_list.append(node)
                self._X = np.array(self._X)

        print(self.get_method_name(), "imported", self._X.shape)


    def get_reconstructed_adj(self, edge_type=None, node_l=None):
        '''Compute the adjacency matrix from the learned embedding

        Returns:
            A numpy array of size #nodes * #nodes containing the reconstructed adjacency matrix.
        '''
        if self._method_name == "LINE":
            reconstructed_adj = np.divide(1, 1 + np.exp(-np.matmul(self._X, self._X.T)))

        elif self._method_name == "node2vec":
            reconstructed_adj = self.softmax(np.dot(self._X, self._X.T))

        elif self._method_name == "source_target_graph_embedding":
            reconstructed_adj = pairwise_distances(X=self._X[:, 0:int(self._d / 2)],
                                                   Y=self._X[:, int(self._d / 2):self._d],
                                                   metric="euclidean", n_jobs=8)
            reconstructed_adj = np.exp(-2.0 * reconstructed_adj)

        elif self._method_name == "HOPE":
            reconstructed_adj = np.matmul(self._X[:, 0:int(self._d / 2)], self._X[:, int(self._d / 2):self._d].T)

        if node_l == self.node_list:
            return reconstructed_adj
        elif set(node_l) < set(self.node_list):
            idx = [self.node_list.index(node) for node in node_l]
            return reconstructed_adj[idx, :][:, idx]
        else:
            raise Exception("A node in node_l is not in self.node_list.")


    def softmax(self, X):
        exps = np.exp(X)
        return exps/np.sum(exps, axis=0)

    def get_top_k_predicted_edges(self, edge_type, top_k, node_list=None, training_network=None):
        nodes = self.node_list
        if node_list is not None:
            nodes = [n for n in nodes if n in node_list]

        estimated_adj = self.get_reconstructed_adj(edge_type=edge_type, node_l=nodes)
        np.fill_diagonal(estimated_adj, 0)

        if training_network is not None:
            training_adj = training_network.get_adjacency_matrix(edge_types=[edge_type], node_list=nodes)
            assert estimated_adj.shape == training_adj.shape
            rows, cols = training_adj.nonzero()
            estimated_adj[rows, cols] = 0

        top_k_indices = largest_indices(estimated_adj, top_k, smallest=False)
        top_k_pred_edges = [(node_list[x[0]], node_list[x[1]], estimated_adj[x[0], x[1]]) for x in zip(*top_k_indices)]

        return top_k_pred_edges


    def predict(self, X):
        reconstructed_adj = self.get_reconstructed_adj()
        node_set = set(self.node_list)

        X_u_inx = [u for u, v in X if u in node_set and v in node_set]
        X_v_inx = [v for u, v in X if u in node_set and v in node_set]

        if len(X_u_inx) == X.shape[0] and len(X_v_inx) == X.shape[0]:
            y_pred = reconstructed_adj[X_u_inx, X_v_inx]
        else:
            y_pred = []
            for u, v in X:
                if u in node_set and v in node_set:
                    y_pred.append(reconstructed_adj[self.node_list.index(u), self.node_list.index(v)])
                else:
                    y_pred.append(0.0)

        y_pred = np.array(y_pred, dtype=np.float).reshape((-1, 1))
        return y_pred

