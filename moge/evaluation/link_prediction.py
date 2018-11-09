import numpy as np

from moge.embedding.static_graph_embedding import StaticGraphEmbedding
from moge.evaluation.metrics import link_prediction_score
from moge.evaluation.utils import mask_test_edges, split_train_test_edges
from moge.network.heterogeneous_network import HeterogeneousNetwork


def evaluate_top_k_link_prediction(top_k, network:HeterogeneousNetwork, graph_emb:StaticGraphEmbedding,
                                   edge_type, node_list, is_directed=True,
                                   test_frac=0.01, val_frac=0., seed=0,
                                   train_embedding=True,
                                   metrics=["precision", "recall"]):
    """
    Evaluates a
    """

    network_train, test_edges, val_edges = split_train_test_edges(network, node_list,
                                                                  edge_types=[edge_type],
                                                                  test_frac=test_frac, val_frac=val_frac,
                                                                  seed=seed, verbose=True)


    print("test_edges:", len(test_edges))

    if train_embedding:
        graph_emb.learn_embedding(network)
    estimated_adj = graph_emb.get_reconstructed_adj(edge_type=edge_type, node_l=node_list)
    train_edges_idx = [(node_list.index(u), node_list.index(v)) for u,v in network_train.G.edges()]

    # evaluate precision/recall at top k predictions, excluding training edges
    top_k_pred_edges_ind = select_top_k_link_predictions(top_k, estimated_adj, train_edges_idx)
    top_k_pred_edges = [x for x in zip(*top_k_pred_edges_ind)]

    test_edges = [x for x in zip(test_edges[:, 0], test_edges[:, 1])]
    score = link_prediction_score(test_edges, top_k_pred_edges, directed=is_directed, metrics=metrics)

    return score


def select_top_k_link_predictions(top_k, estimated_adj, excluding_edges_idx=None, smallest=False):
    # Exclude edges already seen at training time
    if excluding_edges_idx is not None:
        assert excluding_edges_idx[0, 0] is int
        estimated_adj[excluding_edges_idx[:, 0], excluding_edges_idx[:, 1]] = 0

    top_k_indices = largest_indices(estimated_adj, top_k, smallest=smallest)

    return top_k_indices


def evaluate_random_link_prediction(top_k, network:HeterogeneousNetwork, edge_type, node_list,
                                    test_frac=0.01, val_frac=0., seed=0,
                                    metrics=["precision", "recall"]):
    if edge_type == 'd':
        is_directed = True
        true_adj = network.get_adjacency_matrix(edge_types=edge_type, node_list=node_list)
        adj_train, train_edges, \
        val_edges, test_edges = mask_test_edges(true_adj,
                                                is_directed=is_directed,
                                                test_frac=test_frac, val_frac=val_frac,
                                                prevent_disconnect=True, seed=seed, verbose=False)

    elif edge_type == 'u':
        is_directed = False
        true_adj = network.get_adjacency_matrix(edge_types=edge_type, node_list=node_list)
        adj_train, train_edges, \
        val_edges, test_edges = mask_test_edges(true_adj,
                                                is_directed=is_directed,
                                                test_frac=test_frac, val_frac=val_frac,
                                                prevent_disconnect=True, seed=seed, verbose=False)
    else:
        raise Exception("Unsupported edge_type" + edge_type)

    top_k_pred_edges_ind = select_random_link_predictions(top_k, adj_train, train_edges, seed)
    top_k_pred_edges = [x for x in zip(*top_k_pred_edges_ind)]

    test_edges = [x for x in zip(test_edges[:, 0], test_edges[:, 1])]
    score = link_prediction_score(test_edges, top_k_pred_edges, directed=is_directed, metrics=metrics)

    return score


def select_random_link_predictions(top_k, estimated_adj, excluding_edges, seed=0):
    np.random.seed(seed)
    random_adj = np.random.rand(*estimated_adj.shape)
    random_adj[excluding_edges[:, 0], excluding_edges[:, 1]] = 0

    top_k_indices = largest_indices(random_adj, top_k)

    return top_k_indices


def largest_indices(array, k, smallest=False):
    """Returns the k largest indices from a numpy array using partition O(n + k lg k) """
    flat = array.flatten()
    indices = np.argpartition(flat, -k)[-k:]
    if smallest:
        indices = indices[np.argsort(flat[indices])]
    else:
        indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, array.shape)













