import numpy as np
import random
import scipy.sparse as sp
import networkx as nx
from networkx.algorithms.bridges import bridges as bridges
from networkx.algorithms.chains import chain_decomposition

def getRandomEdgePairs(sparse_adj_matrix, node_list, sample_ratio=0.01, return_indices=False):
    rows, cols = sparse_adj_matrix.nonzero()
    num_pairs = int(sample_ratio * len(rows))

    np.random.seed(random.randint(0, 1000000))
    rand_indices = np.random.choice(range(len(rows)), size=num_pairs, replace=False)
    if return_indices:
        return [rows[i] for i in rand_indices], [cols[i] for i in rand_indices]
    else:
        return [(node_list[rows[i]], node_list[cols[i]]) for i in rand_indices]

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


# Perform train-test split
# Takes in adjacency matrix in sparse format (from a directed graph)
# Returns: adj_train, train_edges, val_edges, val_edges_false,
# test_edges, test_edges_false
def mask_test_edges_directed(adj, test_frac=.1, val_frac=.05,
                             prevent_disconnect=True, verbose=False, false_edge_sampling='iterative'):
    if verbose == True:
        print('preprocessing...')


    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    # Convert to networkx graph to calc num. weakly connected components
    g = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph())
    orig_num_wcc = nx.number_weakly_connected_components(g)

    adj_tuple = sparse_to_tuple(adj)  # (coords, values, shape)
    edges = adj_tuple[0]  # List of ALL edges (either direction)
    edge_pairs = [(edge[0], edge[1]) for edge in edges]  # store edges as list of tuples (from_node, to_node)

    num_test = int(np.floor(edges.shape[0] * test_frac))  # controls how large the test set should be
    num_val = int(np.floor(edges.shape[0] * val_frac))  # controls how alrge the validation set should be
    num_train = len(edge_pairs) - num_test - num_val  # num train edges

    all_edge_set = set(edge_pairs)
    train_edges = set(edge_pairs)  # init train_edges to have all edges
    test_edges = set()  # init test_edges as empty set
    val_edges = set()  # init val edges as empty set

    ### ---------- TRUE EDGES ---------- ###
    # Shuffle and iterate over all edges
    np.random.shuffle(edge_pairs)

    # get initial bridge edges
    bridge_edges = set(bridges(g.to_undirected()))

    if verbose:
        print('creating true edges...')

    for ind, edge in enumerate(edge_pairs):
        node1, node2 = edge[0], edge[1]

        # Recalculate bridges every ____ iterations to relatively recent
        if ind % 10000 == 0 and prevent_disconnect:
            bridge_edges = set(bridges(g.to_undirected()))

            # Don't sample bridge edges to increase likelihood of staying connected
        if (node1, node2) in bridge_edges or (node2, node1) in bridge_edges:
            continue

        # If removing edge would disconnect the graph, backtrack and move on
        g.remove_edge(node1, node2)
        if prevent_disconnect == True:
            if not nx.is_weakly_connected(g):
                g.add_edge(node1, node2)
                continue

        # Fill test_edges first
        if len(test_edges) < num_test:
            test_edges.add(edge)
            train_edges.remove(edge)
            if len(test_edges) % 10000 == 0 and verbose == True:
                print('Current num test edges: ', len(test_edges))


        # Then, fill val_edges
        elif len(val_edges) < num_val:
            val_edges.add(edge)
            train_edges.remove(edge)
            if len(val_edges) % 10000 == 0 and verbose == True:
                print('Current num val edges: ', len(val_edges))


        # Both edge lists full --> break loop
        elif len(test_edges) == num_test and len(val_edges) == num_val:
            break

    # Check that enough test/val edges were found
    if (len(val_edges) < num_val or len(test_edges) < num_test):
        print("WARNING: not enough removable edges to perform full train-test split!")

        print("Num. (test, val) edges requested: (", num_test, ", ", num_val, ")")

        print("Num. (test, val) edges returned: (", len(test_edges), ", ", len(val_edges), ")")


    # Print stats for largest remaining WCC
    print('Num WCC: ', nx.number_weakly_connected_components(g))

    largest_wcc_set = max(nx.weakly_connected_components(g), key=len)
    largest_wcc = g.subgraph(largest_wcc_set)
    print('Largest WCC num nodes: ', largest_wcc.number_of_nodes())

    print('Largest WCC num edges: ', largest_wcc.number_of_edges())


    if prevent_disconnect == True:
        assert nx.number_weakly_connected_components(g) == orig_num_wcc

    # Fraction of edges with both endpoints in largest WCC
    def frac_edges_in_wcc(edge_set):
        num_wcc_contained_edges = 0.0
        num_total_edges = 0.0
        for edge in edge_set:
            num_total_edges += 1
            if edge[0] in largest_wcc_set and edge[1] in largest_wcc_set:
                num_wcc_contained_edges += 1
        frac_in_wcc = num_wcc_contained_edges / num_total_edges
        return frac_in_wcc

    # Check what percentage of edges have both endpoints in largest WCC
    print('Fraction of train edges with both endpoints in L-WCC: ', frac_edges_in_wcc(train_edges))

    print('Fraction of test edges with both endpoints in L-WCC: ', frac_edges_in_wcc(test_edges))

    print('Fraction of val edges with both endpoints in L-WCC: ', frac_edges_in_wcc(val_edges))


    # Ignore edges with endpoint not in largest WCC
    print('Removing edges with either endpoint not in L-WCC from train-test split...')

    train_edges = {edge for edge in train_edges if edge[0] in largest_wcc_set and edge[1] in largest_wcc_set}
    test_edges = {edge for edge in test_edges if edge[0] in largest_wcc_set and edge[1] in largest_wcc_set}
    val_edges = {edge for edge in val_edges if edge[0] in largest_wcc_set and edge[1] in largest_wcc_set}

    ### ---------- FALSE EDGES ---------- ###

    # Initialize empty sets
    # train_edges_false = set()
    # test_edges_false = set()
    # val_edges_false = set()



    ### ---------- FINAL DISJOINTNESS CHECKS ---------- ###
    if verbose == True:
        print('final checks for disjointness...')


    # assert: false_edges are actually false (not in all_edge_tuples)
    # assert test_edges_false.isdisjoint(all_edge_set)
    # assert val_edges_false.isdisjoint(all_edge_set)
    # assert train_edges_false.isdisjoint(all_edge_set)
    #
    # # assert: test, val, train false edges disjoint
    # assert test_edges_false.isdisjoint(val_edges_false)
    # assert test_edges_false.isdisjoint(train_edges_false)
    # assert val_edges_false.isdisjoint(train_edges_false)

    # assert: test, val, train positive edges disjoint
    assert val_edges.isdisjoint(train_edges)
    assert test_edges.isdisjoint(train_edges)
    assert val_edges.isdisjoint(test_edges)

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