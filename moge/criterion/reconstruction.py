import numpy as np
from moge.evaluation.utils import getRandomEdgePairs
from moge.model.static_graph_embedding import BaseGraphEmbedding

from moge.network.multi_digraph import MultiDigraphNetwork

flatten = lambda l: [item for sublist in l for item in sublist]


def evaluateStaticGraphReconstruction(network: MultiDigraphNetwork, graph_emb: BaseGraphEmbedding,
                                      edge_type, modalities=None, train_embedding=False,
                                      node_list=None, sample_ratio=0.1, seed=0):
    if node_list is not None:
        node_list = node_list
    elif modalities != None:
        node_list = flatten([network.modality_to_nodes[modality] for modality in modalities])
    else:
        node_list = network.node_list

    true_adj_matrix = network.get_adjacency_matrix(edge_types=edge_type, node_list=node_list)

    if sample_ratio < 1.0:
        eval_edge_rows, eval_edge_cols = getRandomEdgePairs(true_adj_matrix, node_list=node_list,
                                             sample_ratio=sample_ratio, return_indices=True, seed=seed)
    else:
        eval_edge_rows, eval_edge_cols = true_adj_matrix.nonzero()
    print("Sampling", len(eval_edge_rows), "edges to be evaluated.")

    true_edges = true_adj_matrix[eval_edge_rows, eval_edge_cols]

    if ~hasattr(graph_emb, "_X") and train_embedding:  # If graph model isn't trained
        graph_emb.learn_embedding(network.G)

    estimated_edges = graph_emb.get_reconstructed_adj(edge_type=edge_type)[eval_edge_rows, eval_edge_cols]

    norm = np.linalg.norm(true_edges-estimated_edges)
    avg = np.average(np.abs(true_edges - estimated_edges))

    return norm, avg


