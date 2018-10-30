from abc import ABCMeta

import numpy as np


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

        return self._X


    def get_reconstructed_adj(self, edge_type=None, node_l=None):
        '''Compute the adjacency matrix from the learned embedding

        Returns:
            A numpy array of size #nodes * #nodes containing the reconstructed adjacency matrix.
        '''
        if self._method_name == "LINE":
            return np.divide(1, 1 + np.power(np.e, -np.matmul(self._X, self._X.T)))
        elif self._method_name == "node2vec":
            return self.softmax(np.dot(self._X, self._X.T))

    def softmax(self, X):
        exps = np.exp(X)
        return exps/np.sum(exps, axis=0)

    def predict(self, X):
        reconstructed_adj = self.get_reconstructed_adj()

        y_pred = []
        for u, v in X:
            if u in self.node_list and v in self.node_list:
                y_pred.append(reconstructed_adj[self.node_list.index(u), self.node_list.index(v)])
            else:
                y_pred.append(0.0)

        y_pred = np.array(y_pred, dtype=np.float).reshape((-1, 1))
        return y_pred

