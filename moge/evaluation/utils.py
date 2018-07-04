import numpy as np

def getRandomEdgePairs(sparse_adj_matrix, node_list, sample_ratio=0.01):
    rows, cols = sparse_adj_matrix.nonzero(0)
    num_pairs = int(sample_ratio * len(rows))

    rand_indices = np.random.choice(range(len(rows)), size=num_pairs, replace=False)
    return [(node_list[rows[i]], node_list[cols[i]]) for i in rand_indices]