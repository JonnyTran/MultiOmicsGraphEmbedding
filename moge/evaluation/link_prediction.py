import numpy as np
import networkx as nx
from moge.network.heterogeneous_network import HeterogeneousNetwork
from moge.embedding.dual_graph_embedding import SourceTargetGraphEmbedding
from moge.evaluation.utils import mask_test_edges

def evaluateStaticLinkPrediction(network:HeterogeneousNetwork, graph_emb:SourceTargetGraphEmbedding,
                                 edge_type, test_frac=0.15, val_frac=0., seed=0):

    if edge_type == 'd':

        true_adj = network.get_regulatory_edges_adjacency()
        adj_train, train_edges, \
        val_edges, test_edges = mask_test_edges(true_adj,
                                                is_directed=True,
                                                test_frac=test_frac, val_frac=val_frac,
                                                prevent_disconnect=True, seed=seed, verbose=False)
        network.set_regulatory_edges_training_adjacency(adj_train)

    elif edge_type == 'u':

        true_adj = network.get_node_similarity_adjacency()
        adj_train, train_edges, \
        val_edges, test_edges = mask_test_edges(true_adj,
                                                is_directed=True,
                                                test_frac=test_frac, val_frac=val_frac,
                                                prevent_disconnect=True, seed=seed, verbose=False)
        network.set_node_similarity_training_adjacency(adj_train)
    else:
        raise Exception("Unsupported edge_type" + edge_type)

    graph_emb.learn_embedding(network)

    # Testing reconstruction of unseen edges by evaluating the true vs predicted test edges and get the norm of the difference
    true_edges = true_adj[test_edges[:, 0], test_edges[:, 1]]
    estimated_edges = graph_emb.get_reconstructed_adj(edge_type=edge_type)[test_edges[:, 0], test_edges[:, 1]]
    diff = true_edges - estimated_edges

    norm = np.linalg.norm(diff)
    avg = np.average(diff)

    # TODO evaluate precision/recall at top k predictions

    return norm, avg