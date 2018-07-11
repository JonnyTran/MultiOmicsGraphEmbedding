from moge.network.heterogeneous_network import HeterogeneousNetwork
from moge.evaluation.utils import getRandomEdgePairs
import numpy as np
import networkx as nx

flatten = lambda l: [item for sublist in l for item in sublist]


def evaluateStaticGraphReconstruction(network:HeterogeneousNetwork, graph_emb, edge_type, modalities=None,
                                      node_list=None, sample_ratio=0.1):

    if node_list != None:
        node_list = node_list
    elif modalities != None:
        node_list = flatten([network.nodes[modality] for modality in modalities])
    else:
        node_list = network.all_nodes

    if edge_type == "u":
        true_adj_matrix = network.get_node_similarity_adjacency(node_list)
    elif edge_type == "d":
        true_adj_matrix = network.get_regulatory_edges_adjacency(node_list)
    else:
        true_adj_matrix = nx.adjacency_matrix(network.G, nodelist=node_list)

    if sample_ratio < 1.0:
        eval_edge_rows, eval_edge_cols  = getRandomEdgePairs(true_adj_matrix, node_list=node_list,
                                             sample_ratio=sample_ratio, return_indices=True)
    else:
        eval_edge_rows, eval_edge_cols = true_adj_matrix.nonzero()
    print("Sampling", len(eval_edge_rows), "edges to be evaluated.")

    true_edges = true_adj_matrix[eval_edge_rows, eval_edge_cols]
    estimated_edges = graph_emb.get_reconstructed_adj(edge_type=edge_type)[eval_edge_rows, eval_edge_cols]

    norm = np.linalg.norm(true_edges-estimated_edges)
    avg = np.average(true_edges - estimated_edges)

    return norm, avg


