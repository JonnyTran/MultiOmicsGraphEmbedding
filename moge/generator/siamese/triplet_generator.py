import networkx as nx
import numpy as np
import tensorflow as tf

from moge.generator.sampled_generator import SampledDataGenerator
from moge.generator.siamese.pairs_generator import DIRECTED_EDGE, UNDIRECTED_EDGE, \
    UNDIRECTED_NEG_EDGE, IS_DIRECTED, IS_UNDIRECTED
from moge.network.heterogeneous import HeterogeneousNetwork, EPSILON


def sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


class SampledTripletDataGenerator(SampledDataGenerator):
    def __init__(self, network: HeterogeneousNetwork, variables=None, targets=None, weighted=False, batch_size=1,
                 replace=True, seed=0, verbose=True, **kwargs):
        super(SampledTripletDataGenerator, self).__init__(network=network, compression_func=compression_func,
                                                          n_steps=n_steps, directed=directed_proba, weighted=weighted,
                                                          batch_size=batch_size, replace=replace, seed=seed,
                                                          verbose=verbose, maxlen=maxlen, padding=padding,
                                                          truncating=truncating, sequence_to_matrix=sequence_to_matrix,
                                                          tokenizer=tokenizer)

    def __getitem__(self, item):
        sampled_edges = []
        while len(sampled_edges) < self.batch_size:
            sampled_node = np.random.choice(self.node_list, size=1, replace=True,
                                            p=self.node_sampling_freq)
            sampled_triplet = self.sample_triplet_from_node(sampled_node[0])
            if sampled_triplet is not None:
                sampled_edges.append(sampled_triplet)
            else:
                continue
        X, y = self.__getdata__(sampled_edges)

        return X, y

    def sample_triplet_from_node(self, anchor_node):
        edge_type = self.sample_edge_type(self.edge_dict[anchor_node].keys())
        if edge_type == DIRECTED_EDGE:
            pos_sample = next(self.edge_dict[anchor_node][edge_type])  # ((node_u, node_v, edge_type)
            neg_sample = self.get_negative_sampled_edges(anchor_node)

        elif edge_type == UNDIRECTED_EDGE:
            pos_sample = next(self.edge_dict[anchor_node][edge_type])
            neg_sample = next(self.edge_dict[anchor_node][UNDIRECTED_NEG_EDGE]) if UNDIRECTED_NEG_EDGE in \
                                                                                   self.edge_dict[
                                                                                       anchor_node].keys() else self.get_negative_sampled_edges(
                anchor_node)
        else:
            return None

        return (anchor_node, pos_sample[1], neg_sample[1], edge_type)

    def sample_edge_type(self, edge_types):
        if DIRECTED_EDGE in edge_types and UNDIRECTED_EDGE in edge_types:
            edge_type = np.random.choice([DIRECTED_EDGE, UNDIRECTED_EDGE],
                                         p=[self.directed_proba, 1 - self.directed_proba])
        elif DIRECTED_EDGE in edge_types:
            edge_type = DIRECTED_EDGE
        elif UNDIRECTED_EDGE in edge_types:
            edge_type = UNDIRECTED_EDGE
        else:
            return None

        return edge_type

    def __getdata__(self, sampled_nodes):
        'Returns the training data (X, y) tuples given a list of tuple(source_id, target_id, is_directed, edge_weight)'
        X_list = []
        for u, v, w, type in sampled_nodes:
            if type == DIRECTED_EDGE:
                X_list.append((u, v, w, IS_DIRECTED))
            elif type == UNDIRECTED_EDGE:
                X_list.append((u, v, w, IS_UNDIRECTED))
            else:
                raise Exception("Edge type is wrong:" + u + v + w + type)

        # assert self.batch_size == len(X_list)
        X_list = np.array(X_list, dtype="O")

        X = {}
        X["input_seq_i"] = self.get_sequence_data(X_list[:, 0].tolist(), variable_length=False)
        X["input_seq_j"] = self.get_sequence_data(X_list[:, 1].tolist(), variable_length=False)
        X["input_seq_k"] = self.get_sequence_data(X_list[:, 2].tolist(), variable_length=False)
        X["is_directed"] = np.expand_dims(X_list[:, 3], axis=-1)

        y = np.zeros(X_list[:, 3].shape)

        return X, y


class OnlineTripletGenerator(SampledDataGenerator):
    def __init__(self, network: HeterogeneousNetwork, variables=None, targets=None, weighted=False, batch_size=1,
                 replace=True, seed=0, verbose=True, **kwargs):
        super(OnlineTripletGenerator, self).__init__(network=network, weighted=weighted, batch_size=batch_size,
                                                     replace=replace, seed=seed, verbose=verbose, maxlen=maxlen,
                                                     padding=padding, truncating=truncating,
                                                     sequence_to_matrix=sequence_to_matrix, tokenizer=tokenizer)

    def process_sampling_table(self, network):
        graph = nx.compose(network.G, network.G_u)
        self.node_degrees = dict(zip(self.node_list, [0] * len(self.node_list)))

        for node in network.node_list:
            edgelist_bunch = graph.edges(node, data=True)
            self.node_degrees[node] = len(edgelist_bunch)

        self.node_degrees_list = [self.node_degrees[node] for node in self.node_list]
        self.node_sampling_freq = self.compute_node_sampling_freq(self.node_degrees_list,
                                                                  compression=self.compression_func)
        print("# of nodes to sample from (non-zero degree):",
              np.count_nonzero(self.node_sampling_freq)) if self.verbose else None

    def __getitem__(self, item):
        sampled_nodes = np.random.choice(self.node_list, size=self.batch_size, replace=False,
                                         p=self.node_sampling_freq)
        X, y = self.__getdata__(sampled_nodes)

        return X, y

    def load_data(self, return_node_name=False):
        sampled_nodes = self.get_nonzero_nodelist()
        X, y = self.__getdata__(sampled_nodes, return_node_name=return_node_name)

        return X, y

    def __getdata__(self, sampled_nodes, return_node_name=False):
        X = {}
        X["input_seqs"] = self.get_sequence_data(sampled_nodes, variable_length=False)
        sampled_directed_adj = self.sample_random_negative_edges(
            self.network.get_adjacency_matrix(edge_types=["d"], node_list=sampled_nodes),
            sampled_nodes,
            self.negative_sampling_ratio)
        X["labels_directed"] = sampled_directed_adj
        X["labels_undirected"] = self.network.get_adjacency_matrix(edge_types=["u", "u_n"], node_list=sampled_nodes)

        y = np.asarray([self.node_list.index(node) for node in sampled_nodes], dtype=np.int)
        if return_node_name: y = np.asarray(sampled_nodes, dtype="O")

        return X, y

    def sample_directed_negative_edges(self, pos_adj, sampled_nodes, negative_sampling_ratio):
        """
        Samples a number of negative edges with context to the number of positive edges in the adjacency matrix.
        For each node, if n is the number of its positive connections, this function will sample n*k negative connections,
        based on the unigram distribution of the node degrees, while excluding accidental hits of positive connections.

        :param pos_adj: a sparse csr_matrix of shape [batch_size, batch_size] representing a sampled adjacency matrix containing only positive interactions
        :return: a lil sparse matrix containing both positive interactions and sampled negative interactions
        """
        node_degrees_list = [self.node_degrees[node] for node in sampled_nodes]

        sampled_adj = pos_adj.tolil()
        for idx, node in enumerate(sampled_nodes):
            _, pos_nodes = pos_adj[idx].nonzero()
            node_neg_sample_count = min(int(len(pos_nodes) * negative_sampling_ratio),
                                        int(pos_adj.shape[1] * 0.2))
            if node_neg_sample_count > 0:
                node_degrees = [degree if (id not in pos_nodes and id != idx) else 0 for id, degree in
                                enumerate(node_degrees_list)]  # Prevent accidental candidate sampling
                sample_neg_indices = np.random.choice(range(len(sampled_nodes)), node_neg_sample_count, replace=False,
                                                      p=self.compute_node_sampling_freq(node_degrees,
                                                                                        compression="linear"))
                sampled_adj[idx, sample_neg_indices] = EPSILON
        assert sampled_adj.count_nonzero() > pos_adj.count_nonzero(), "Did not add any sampled negative edges {} > {}".format(
            sampled_adj.count_nonzero(),
            pos_adj.count_nonzero())

        return sampled_adj

    def sample_random_negative_edges(self, pos_adj, sampled_nodes, negative_sampling_ratio):
        """
        This samples a number of negative edges in proportion to the number of positive edges in the adjacency matrix,
        by sampling uniformly random edges.

        :param pos_adj: a sparse csr_matrix of shape [batch_size, batch_size] representing a sampled adjacency matrix containing only positive interactions
        :return: a sparse matrix containing both positive interactions and sampled negative interactions
        """
        pos_rows, pos_cols = pos_adj.nonzero()
        Ed_count = len(pos_rows)
        sample_neg_count = min(int(Ed_count * negative_sampling_ratio), np.power(pos_adj.shape[0], 2) * 0.50)

        neg_rows, neg_cols = np.where(pos_adj.todense() == 0)
        sample_indices = np.random.choice(neg_rows.shape[0], sample_neg_count, replace=False)
        pos_neg_adj = pos_adj.tolil()
        pos_neg_adj[neg_rows[sample_indices], neg_cols[sample_indices]] = EPSILON
        assert pos_neg_adj.count_nonzero() > pos_adj.count_nonzero(), "Did not add any sampled negative edges {} > {}".format(
            pos_neg_adj.count_nonzero(), pos_adj.count_nonzero())
        return pos_neg_adj


class OnlineSoftmaxGenerator(OnlineTripletGenerator):
    def __init__(self, network: HeterogeneousNetwork, variables=None, targets=None, weighted=False, batch_size=1,
                 replace=True, seed=0, verbose=True, **kwargs):
        super(OnlineSoftmaxGenerator, self).__init__(network, weighted=weighted, batch_size=batch_size, replace=replace,
                                                     seed=seed, verbose=verbose, maxlen=maxlen, padding=padding,
                                                     truncating=truncating, sequence_to_matrix=sequence_to_matrix,
                                                     tokenizer=tokenizer)

    def __getdata__(self, sampled_nodes):
        X = {}
        X["input_seqs"] = self.get_sequence_data(sampled_nodes, variable_length=False)
        sampled_directed_adj = self.sample_directed_negative_edges(
            self.network.get_adjacency_matrix(edge_types=["d"], node_list=sampled_nodes), sampled_nodes)
        X["labels_directed"] = sampled_directed_adj
        X["labels_undirected"] = self.network.get_adjacency_matrix(edge_types=["u", "u_n"], node_list=sampled_nodes)

        y = np.zeros(X["input_seqs"].shape[0])  # Dummy vector
        return X, y
