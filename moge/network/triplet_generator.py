import numpy as np

from moge.network.edge_generator import SampledDataGenerator, DIRECTED_EDGE_TYPE, UNDIRECTED_EDGE_TYPE, \
    UNDIRECTED_NEG_EDGE_TYPE, IS_DIRECTED, IS_UNDIRECTED
from moge.network.heterogeneous_network import HeterogeneousNetwork, EPSILON


def sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return indices, coo.data.astype(np.float), coo.shape

class SampledTripletDataGenerator(SampledDataGenerator):
    def __init__(self, network: HeterogeneousNetwork,
                 batch_size=1, directed_proba=0.5, negative_sampling_ratio=3, n_steps=500, compression_func="log",
                 maxlen=1400, padding='post', truncating='post', sequence_to_matrix=False,
                 shuffle=True, seed=0, verbose=True):
        super().__init__(network=network,
                         batch_size=batch_size, negative_sampling_ratio=negative_sampling_ratio, n_steps=n_steps,
                         directed_proba=directed_proba, compression_func=compression_func,
                         maxlen=maxlen, padding=padding, truncating=truncating, sequence_to_matrix=sequence_to_matrix,
                         shuffle=shuffle, seed=seed, verbose=verbose)

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
        X, y = self.__data_generation(sampled_edges)

        return X, y

    def sample_triplet_from_node(self, anchor_node):
        edge_type = self.sample_edge_type(self.edge_dict[anchor_node].keys())
        if edge_type == DIRECTED_EDGE_TYPE:
            pos_sample = next(self.edge_dict[anchor_node][edge_type]) # ((node_u, node_v, edge_type)
            neg_sample = self.get_negative_sampled_edges(anchor_node)

        elif edge_type == UNDIRECTED_EDGE_TYPE:
            pos_sample = next(self.edge_dict[anchor_node][edge_type])
            neg_sample = next(self.edge_dict[anchor_node][UNDIRECTED_NEG_EDGE_TYPE]) if UNDIRECTED_NEG_EDGE_TYPE in self.edge_dict[anchor_node].keys() else self.get_negative_sampled_edges(anchor_node)
        else:
            return None

        return (anchor_node, pos_sample[1], neg_sample[1], edge_type)

    def sample_edge_type(self, edge_types):
        if DIRECTED_EDGE_TYPE in edge_types and UNDIRECTED_EDGE_TYPE in edge_types:
            edge_type = np.random.choice([DIRECTED_EDGE_TYPE, UNDIRECTED_EDGE_TYPE], p=[self.directed_proba, 1-self.directed_proba])
        elif DIRECTED_EDGE_TYPE in edge_types:
            edge_type = DIRECTED_EDGE_TYPE
        elif UNDIRECTED_EDGE_TYPE in edge_types:
            edge_type = UNDIRECTED_EDGE_TYPE
        else:
            return None

        return edge_type


    def __data_generation(self, sampled_edges):
        'Returns the training data (X, y) tuples given a list of tuple(source_id, target_id, is_directed, edge_weight)'
        X_list = []
        for u,v,w,type in sampled_edges:
            if type == DIRECTED_EDGE_TYPE:
                X_list.append((u, v, w, IS_DIRECTED))
            elif type == UNDIRECTED_EDGE_TYPE:
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
    def __init__(self, network: HeterogeneousNetwork, batch_size=1, directed_proba=0.5, negative_sampling_ratio=20.0,
                 n_steps=500, compression_func="log", maxlen=2000, padding='post', truncating='post',
                 sequence_to_matrix=False, shuffle=True, seed=0, verbose=True):
        super().__init__(network, batch_size, directed_proba, negative_sampling_ratio, n_steps, compression_func,
                         maxlen, padding, truncating, sequence_to_matrix, shuffle, seed, verbose)

    def __getitem__(self, item):
        sampled_nodes = np.random.choice(self.node_list, size=self.batch_size, replace=False,
                                         p=self.node_sampling_freq)
        X, y = self.__data_generation(sampled_nodes)

        return X, y

    def __data_generation(self, sampled_nodes):
        X = {}
        X["input_seqs"] = self.get_sequence_data(sampled_nodes, variable_length=False)
        sampled_directed_adj = self.sample_directed_negative_edges(self.network.get_adjacency_matrix(edge_types=["d"], node_list=sampled_nodes))
        X["labels_directed"] = sparse_matrix_to_sparse_tensor(sampled_directed_adj)
        X["labels_undirected"] = sparse_matrix_to_sparse_tensor(self.network.get_adjacency_matrix(edge_types=["u", "u_n"], node_list=sampled_nodes))

        y = np.zeros(X["input_seqs"].shape[0]) # Dummy vector
        return X, y

    def sample_directed_negative_edges(self, adj):
        """
        This samples a number of negative edges in proportion to the number of positive edges in the adjacency matrix

        :param adj: a sparse csr_matrix of shape [batch_size, batch_size] representing a sampled adjacency matrix containing only positive interactions
        :return: a sparse matrix containing both positive interactions and sampled negative interactions
        """
        pos_rows, pos_cols = adj.nonzero()
        Ed_count = len(pos_rows)

        neg_rows, neg_cols = np.where(adj.todense() == 0)
        sample_neg_count = int(Ed_count * self.negative_sampling_ratio)
        sample_indices = np.random.choice(neg_rows.shape[0], sample_neg_count, replace=False)
        adj[neg_rows[sample_indices], neg_cols[sample_indices]] = EPSILON

        return adj






