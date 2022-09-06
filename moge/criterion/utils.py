import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse as sp


def get_scalefree_fit_score(degrees, k_power=1, plot=False):
    if len(degrees.shape) > 1:
        degrees = np.ravel(degrees)

    cosine_adj_hist = np.histogram(np.power(degrees, k_power), bins=1000)
    cosine_adj_hist_dist = scipy.stats.rv_histogram(cosine_adj_hist)

    c = np.log10(np.power(degrees, k_power))
    d = np.log10(cosine_adj_hist_dist.pdf(np.power(degrees, k_power)))
    if plot:
        plt.scatter(x=c, y=d, marker='.')
        plt.xlabel("np.log10(k)")
        plt.ylabel("np.log10(P(k))")
        plt.show()
    d_ = d[np.where(c != -np.inf)]
    d_ = d_[np.where(d_ != -np.inf)]
    c_ = c[np.where(d != -np.inf)]
    c_ = c_[np.where(c_ != -np.inf)]
    r_square = np.power(scipy.stats.pearsonr(c_, d_)[0], 2)
    return r_square

def sample_edges(nodes_A, nodes_B, n_edges, edge_type="u_n", edge_weight=1e-8):
    edges_u = np.random.choice(nodes_A, size=n_edges, replace=True)
    edges_v = np.random.choice(nodes_B, size=n_edges, replace=True)

    edge_ebunch = [(u, v, {"type":edge_type, "weight":edge_weight}) for u,v in zip(edges_u, edges_v)]

    return edge_ebunch

def getRandomEdgePairs(sparse_adj_matrix, node_list=None, sample_ratio=0.01, return_indices=True, seed=0):
    rows, cols = sparse_adj_matrix.nonzero()
    num_pairs = int(sample_ratio * len(rows))

    np.random.seed(seed)
    rand_indices = np.random.choice(range(len(rows)), size=num_pairs, replace=False)
    if return_indices:
        return [rows[i] for i in rand_indices], [cols[i] for i in rand_indices]
    elif node_list is not None:
        return [(node_list[rows[i]], node_list[cols[i]]) for i in rand_indices]


def sparse_to_tuple(sparse_mx):
    # Convert sparse matrix to tuple
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    # Get normalized adjacency matrix: A_norm
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def largest_indices(array, k, smallest=False):
    """Returns the k largest indices from a numpy array using partition O(n + k lg k) """
    order = "C"
    flat = np.ravel(array, order=order)
    indices = np.argpartition(flat, -k)[-k:]
    if smallest:
        indices = indices[np.argsort(flat[indices])]
    else:
        indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, array.shape, order=order)
