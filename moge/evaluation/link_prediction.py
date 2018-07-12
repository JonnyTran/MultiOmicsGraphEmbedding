import numpy as np
import networkx as nx
from moge.network.heterogeneous_network import HeterogeneousNetwork
from moge.embedding.dual_graph_embedding import StaticGraphEmbedding
from moge.evaluation.utils import mask_test_edges
from moge.evaluation.metrics import link_prediction_score

def evaluate_top_k_link_prediction(top_k, network:HeterogeneousNetwork, graph_emb:StaticGraphEmbedding,
                                   edge_type, test_frac=0.01, val_frac=0., seed=0, train_embedding=True):
    # TODO Implement modality -specific train-test-split
    if edge_type == 'd':
        is_directed = True
        true_adj = network.get_regulatory_edges_adjacency()
        adj_train, train_edges, \
        val_edges, test_edges = mask_test_edges(true_adj,
                                                is_directed=is_directed,
                                                test_frac=test_frac, val_frac=val_frac,
                                                prevent_disconnect=True, seed=seed, verbose=False)
        network.set_regulatory_edges_training_adjacency(adj_train)

    elif edge_type == 'u':
        is_directed = False
        true_adj = network.get_node_similarity_adjacency()
        adj_train, train_edges, \
        val_edges, test_edges = mask_test_edges(true_adj,
                                                is_directed=is_directed,
                                                test_frac=test_frac, val_frac=val_frac,
                                                prevent_disconnect=True, seed=seed, verbose=False)
        network.set_node_similarity_training_adjacency(adj_train)
    else:
        raise Exception("Unsupported edge_type" + edge_type)

    print("test_edges:", test_edges.shape[0])

    if train_embedding:
        graph_emb.learn_embedding(network)
    estimated_adj = graph_emb.get_reconstructed_adj(edge_type=edge_type)

    # evaluate precision/recall at top k predictions, excluding training edges
    top_k_pred_edges_ind = select_top_k_link_predictions(top_k, estimated_adj, train_edges)
    print("top k predicted edges:", estimated_adj[top_k_pred_edges_ind])
    top_k_pred_edges = [x for x in zip(*top_k_pred_edges_ind)]

    test_edges = [x for x in zip(test_edges[:, 0], test_edges[:, 1])]
    score = link_prediction_score(test_edges, top_k_pred_edges, directed=is_directed)

    return score

def select_top_k_link_predictions(top_k, estimated_adj, excluding_edges):
    # Exclude edges already seen at training time
    estimated_adj[excluding_edges[:, 0], excluding_edges[:, 1]] = 0

    top_k_indices = largest_indices(estimated_adj, top_k)

    return top_k_indices

def select_random_link_predictions(top_k, estimated_adj, excluding_edges):
    pass # TODO


def largest_indices(array, k):
    """Returns the k largest indices from a numpy array using partition O(n + k lg k) """
    flat = array.flatten()
    indices = np.argpartition(flat, -k)[-k:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, array.shape)













