import numpy as np
import networkx as nx
from moge.network.heterogeneous_network import HeterogeneousNetwork
from moge.embedding.dual_graph_embedding_node_SGD import StaticGraphEmbedding
from moge.evaluation.utils import mask_test_edges
from moge.evaluation.metrics import link_prediction_score

def evaluate_top_k_link_prediction(top_k, network:HeterogeneousNetwork, graph_emb:StaticGraphEmbedding,
                                   edge_type, node_list, saved_test_edges=None,
                                   test_frac=0.01, val_frac=0., seed=0,
                                   train_embedding=True,
                                   metrics=["precision", "recall"]):
    """
    Evaluates a
    """
    # TODO Implement modality-specific train-test-split
    if edge_type == 'd':
        is_directed = True
        true_adj = network.get_adjacency_matrix(edge_types=edge_type, node_list=node_list)

        if saved_test_edges is None:
            adj_train, train_edges, \
            val_edges, test_edges = mask_test_edges(true_adj,
                                                    is_directed=is_directed,
                                                    test_frac=test_frac, val_frac=val_frac,
                                                    prevent_disconnect=True, seed=seed, verbose=False)
            network.set_regulatory_edges_training_adjacency(adj_train)
        else:
            adj_train, train_edges, \
            _, _ = mask_test_edges(true_adj,
                                   is_directed=is_directed, test_frac=0.0, val_frac=0.0,
                                   prevent_disconnect=True, seed=seed, verbose=False)

            test_edges = saved_test_edges

    elif edge_type == 'u':
        is_directed = False
        true_adj = network.get_adjacency_matrix(edge_types=edge_type, node_list=node_list)

        if saved_test_edges is None:
            adj_train, train_edges, \
            val_edges, test_edges = mask_test_edges(true_adj,
                                                    is_directed=is_directed,
                                                    test_frac=test_frac, val_frac=val_frac,
                                                    prevent_disconnect=True, seed=seed, verbose=False)
            network.set_node_similarity_training_adjacency(adj_train)
        else:
            adj_train, train_edges, \
            _, _ = mask_test_edges(true_adj, is_directed=is_directed,
                                   test_frac=0.0, val_frac=0.0,
                                   prevent_disconnect=True, seed=seed, verbose=False)
            test_edges = saved_test_edges
    else:
        raise Exception("Unsupported edge_type" + edge_type)

    print("test_edges:", test_edges.shape[0])

    if train_embedding:
        graph_emb.learn_embedding(network)
    estimated_adj = graph_emb.get_reconstructed_adj()

    # evaluate precision/recall at top k predictions, excluding training edges
    top_k_pred_edges_ind = select_top_k_link_predictions(top_k, estimated_adj, train_edges)
    # print("top k predicted edges:", estimated_adj[top_k_pred_edges_ind[0]])
    top_k_pred_edges = [x for x in zip(*top_k_pred_edges_ind)]

    test_edges = [x for x in zip(test_edges[:, 0], test_edges[:, 1])]
    score = link_prediction_score(test_edges, top_k_pred_edges, directed=is_directed, metrics=metrics)

    return score


def select_top_k_link_predictions(top_k, estimated_adj, excluding_edges, smallest=False):
    # Exclude edges already seen at training time
    estimated_adj[excluding_edges[:, 0], excluding_edges[:, 1]] = 0

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













