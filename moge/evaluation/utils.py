import numpy as np
import random
import scipy.sparse as sp
import networkx as nx
import copy

from moge.network.heterogeneous_network import HeterogeneousNetwork
from moge.embedding.dual_graph_embedding_node_SGD import ImportedGraphEmbedding


def getRandomEdgePairs(sparse_adj_matrix, node_list=None, sample_ratio=0.01, return_indices=True, seed=0):
    rows, cols = sparse_adj_matrix.nonzero()
    num_pairs = int(sample_ratio * len(rows))

    np.random.seed(seed)
    rand_indices = np.random.choice(range(len(rows)), size=num_pairs, replace=False)
    if return_indices:
        return [rows[i] for i in rand_indices], [cols[i] for i in rand_indices]
    elif node_list is not None:
        return [(node_list[rows[i]], node_list[cols[i]]) for i in rand_indices]

def split_train_test_network(network:HeterogeneousNetwork, node_list, edge_types=["u", "d"],
                             test_frac=.1, val_frac=.05, seed=0):
    network_train = network
    val_edges_dict = {}
    test_edges_dict = {}

    for edge_type in edge_types:
        if edge_type == 'd':
            is_directed = True
        else:
            is_directed = False

        adj_train, train_edges, \
        val_edges, test_edges = mask_test_edges(network.get_adjacency_matrix(edge_type, node_list=node_list),
                                                is_directed=is_directed,
                                                test_frac=test_frac, val_frac=val_frac, seed=seed)
        edge_list = [(node_list[edge[0]], node_list[edge[1]]) for edge in train_edges]
        network_train.remove_edges_from(edge_list)
        print("Removed", len(edge_list), edge_type, "edges")
        val_edges_dict[edge_type] = val_edges
        test_edges_dict[edge_type] = test_edges

    return network_train, val_edges_dict, test_edges_dict

# Convert sparse matrix to tuple
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

# Get normalized adjacency matrix: A_norm
def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def mask_test_edges(adj, is_directed=True, test_frac=.1, val_frac=.05,
                    prevent_disconnect=True, only_largest_wcc=False, seed=0, verbose=False):
    """
    Perform train-test split of the adjancency matrix and return the train-set and test-set edgelist (indices
    instead of node label). Node sampling of the testing set is after excluding bridges edges to prevent disconnect
    (implemented for undirected graph).

    :param adj: adjacency matrix in sparse format
    :param is_directed:
    :param test_frac:
    :param val_frac:
    :param prevent_disconnect:
    :param only_largest_wcc:
    :param seed:
    :param verbose:
    :return:
    """
    if verbose == True:
        print('preprocessing...')

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    # Convert to networkx graph to calc num. weakly connected components
    g = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph() if is_directed else nx.Graph())
    orig_num_wcc = nx.number_weakly_connected_components(g) if is_directed else nx.number_connected_components(g)
    adj_tuple = sparse_to_tuple(adj)  # (coords, values, shape)
    edges = adj_tuple[0]  # List of ALL edges (either direction)
    edge_pairs = [(edge[0], edge[1]) for edge in edges]  # store edges as list of tuples (from_node, to_node)

    num_test = int(np.floor(edges.shape[0] * test_frac))  # controls how large the test set should be
    num_val = int(np.floor(edges.shape[0] * val_frac))  # controls how alrge the validation set should be
    num_train = len(edge_pairs) - num_test - num_val  # num train edges

    ### ---------- TRUE EDGES ---------- ###
    # Shuffle and iterate over all edges

    if verbose:
        print('creating true edges...')

    # Add MST edges to train_edges, to exclude bridge edges from the test and validation set
    mst_edges = set(nx.minimum_spanning_tree(g.to_undirected() if is_directed else g).edges())
    train_edges = set([pair for pair in edge_pairs if
                       (pair[0], pair[1]) in mst_edges or (pair[0], pair[1])[::-1] in mst_edges])
    if verbose: print("edges in MST:", len(train_edges))

    all_edge_set = [pair for pair in edge_pairs if pair not in train_edges]
    np.random.seed(seed)
    np.random.shuffle(all_edge_set)
    train_edges = list(train_edges)

    test_edges = all_edge_set[0 : num_test]
    val_edges = all_edge_set[num_test : num_test+num_val]
    train_edges.extend(all_edge_set[num_test+num_val:])

    # Remove edges from g to test connected-ness
    g.remove_edges_from(test_edges)
    g.remove_edges_from(val_edges)

    # Check that enough test/val edges were found
    if (len(val_edges) < num_val or len(test_edges) < num_test):
        print("WARNING: not enough removable edges to perform full train-test split!")
        print("Num. (test, val) edges requested: (", num_test, ", ", num_val, ")")
        print("Num. (test, val) edges returned: (", len(test_edges), ", ", len(val_edges), ")")

    if prevent_disconnect == True:
        assert nx.number_weakly_connected_components(g) if is_directed else nx.number_connected_components(g) == orig_num_wcc

    # Print stats for largest remaining WCC
    if verbose:
        print('Num WCC: ', nx.number_weakly_connected_components(g) if is_directed else nx.number_connected_components(g))
        largest_wcc_set = max(nx.weakly_connected_components(g) if is_directed else nx.connected_components(g), key=len)
        largest_wcc = g.subgraph(largest_wcc_set)
        print('Largest WCC num nodes: ', largest_wcc.number_of_nodes())
        print('Largest WCC num edges: ', largest_wcc.number_of_edges())


    # Fraction of edges with both endpoints in largest WCC
    def frac_edges_in_wcc(edge_set):
        if len(edge_set) == 0:
            return "N/A"
        num_wcc_contained_edges = 0.0
        num_total_edges = 0.0
        for edge in edge_set:
            num_total_edges += 1
            if edge[0] in largest_wcc_set and edge[1] in largest_wcc_set:
                num_wcc_contained_edges += 1
        frac_in_wcc = num_wcc_contained_edges / num_total_edges
        return frac_in_wcc

    # Check what percentage of edges have both endpoints in largest WCC
    if verbose:
        print('Fraction of train edges with both endpoints in L-WCC: ', frac_edges_in_wcc(train_edges))
        print('Fraction of test edges with both endpoints in L-WCC: ', frac_edges_in_wcc(test_edges))
        print('Fraction of val edges with both endpoints in L-WCC: ', frac_edges_in_wcc(val_edges))


    # Ignore edges with endpoint not in largest WCC
    if only_largest_wcc:
        print('Removing edges with either endpoint not in L-WCC from train-test split...')
        train_edges = {edge for edge in train_edges if edge[0] in largest_wcc_set and edge[1] in largest_wcc_set}
        test_edges = {edge for edge in test_edges if edge[0] in largest_wcc_set and edge[1] in largest_wcc_set}
        val_edges = {edge for edge in val_edges if edge[0] in largest_wcc_set and edge[1] in largest_wcc_set}


    ### ---------- FINAL DISJOINTNESS CHECKS ---------- ###
    if verbose == True:
        print('final checks for disjointness...')


    # assert: test, val, train positive edges disjoint
    assert set(val_edges).isdisjoint(set(train_edges))
    assert set(test_edges).isdisjoint(set(train_edges))
    assert set(val_edges).isdisjoint(set(test_edges))

    if verbose == True:
        print('creating adj_train...')


    # Re-build adj matrix using remaining graph
    adj_train = nx.adjacency_matrix(g)

    # Convert edge-lists to numpy arrays
    train_edges = np.array([list(edge_tuple) for edge_tuple in train_edges])
    val_edges = np.array([list(edge_tuple) for edge_tuple in val_edges])
    test_edges = np.array([list(edge_tuple) for edge_tuple in test_edges])

    if verbose == True:
        print('Done with train-test split!')
        print('Num train edges (true, ): (', train_edges.shape[0], ', ')
        print('Num test edges (true, ): (', test_edges.shape[0], ', ')
        print('Num val edges (true, ): (', val_edges.shape[0], ', ')

    # Return final edge lists (edges can go either direction!)
    return adj_train, train_edges, \
           val_edges, test_edges