import numpy as np
import random

def getRandomEdgePairs(sparse_adj_matrix, node_list, sample_ratio=0.01, return_indices=False):
    rows, cols = sparse_adj_matrix.nonzero()
    num_pairs = int(sample_ratio * len(rows))

    np.random.seed(random.randint(0, 1000000))
    rand_indices = np.random.choice(range(len(rows)), size=num_pairs, replace=False)
    if return_indices:
        return [rows[i] for i in rand_indices], [cols[i] for i in rand_indices]
    else:
        return [(node_list[rows[i]], node_list[cols[i]]) for i in rand_indices]


def split_graph_train_test(network:HeterogeneousNetwork, ):
    pass