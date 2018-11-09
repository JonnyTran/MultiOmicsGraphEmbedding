import numpy as np

from moge.evaluation.metrics import link_prediction_score
from moge.evaluation.utils import mask_test_edges, split_train_test_edges
from moge.network.heterogeneous_network import HeterogeneousNetwork


def evaluate_top_k_link_pred(embedding, network_train, network_test, node_list, edge_type="d", top_k=100):
    nodes = embedding.node_list
    nodes = [node for node in nodes if node in node_list]

    edges_pred = embedding.get_top_k_predicted_edges(edge_type=edge_type, top_k=top_k,
                                                     node_list=nodes,
                                                     training_network=network_train)
    edges_true = network_test.get_edgelist(edge_types=[edge_type], node_list=nodes, inclusive=True)

    results = {}
    results["precision"] = precision(edges_true, edges_pred)
    results["recall"] = recall(edges_true, edges_pred)
    return results


def precision(edges_true, edges_pred):
    if len(edges_pred[0]) > 2:
        edges_pred = [(u, v) for u, v, w in edges_pred]

    true_positives = len(set(edges_true) & set(edges_pred))
    return true_positives / len(edges_pred)


def recall(edges_true, edges_pred):
    if len(edges_pred[0]) > 2:
        edges_pred = [(u, v) for u, v, w in edges_pred]

    true_positives = len(set(edges_true) & set(edges_pred))
    return true_positives / len(edges_true)



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













