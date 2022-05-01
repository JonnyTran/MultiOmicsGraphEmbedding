from typing import List, Union

import networkx as nx
import numpy as np
import pandas as pd
import torch


def one_hot_encoder(x, embed_dim=None):
    """
    Get ont hot embedding of the input tensor.
    Args:
        x: torch.Tensor, input 1-D tensor.
    Returns:
        one_hot: torch.Tensor, one-hot embedding of x.
    """
    ids = x.unique()
    id_dict = dict(list(zip(ids.numpy(), np.arange(len(ids)))))
    one_hot = torch.zeros((len(x), len(ids)))
    for i, u in enumerate(x):
        if id_dict[u.item()] == 4:
            pass
        else:
            one_hot[i][id_dict[u.item()]] = 1

    return one_hot


def edge_dict_sizes(edge_index_dict):
    return {k: v.shape[1] for k, v in edge_index_dict.items()}


def edge_dict_intersection(edge_index_dict_A, edge_index_dict_B):
    inters = {}
    for metapath, edge_index in edge_index_dict_A.items():
        if metapath not in edge_index_dict_B:
            inters[metapath] = 0
            continue

        inters[metapath] = edge_intersection(edge_index, edge_index_dict_B[metapath])

    return inters


def edge_intersection(edge_index_A, edge_index_B, remove_duplicates=True):
    A = pd.DataFrame(edge_index_A.T.numpy(), columns=["source", "target"])
    B = pd.DataFrame(edge_index_B.T.numpy(), columns=["source", "target"])
    int_df = pd.merge(A, B, how='inner', on=["source", "target"], sort=True)
    if remove_duplicates:
        int_df = int_df[~int_df.duplicated()]
    return torch.tensor(int_df.to_numpy().T, dtype=torch.long)


def nonduplicate_indices(edge_index):
    edge_df = pd.DataFrame(edge_index.t().numpy())  # shape: (n_edges, 2)
    return ~edge_df.duplicated(subset=[0, 1])


def merge_node_index(old_node_index, new_node_index):
    merged = {k: [v] for k, v in old_node_index.items()}

    for ntype, new_nodes in new_node_index.items():
        if ntype not in old_node_index:
            merged.setdefault(ntype, []).append(new_nodes)
        else:
            merged.setdefault(ntype, []).append(old_node_index[ntype])
            new_nodes_mask = np.isin(new_nodes, old_node_index[ntype], invert=True)
            merged[ntype].append(new_nodes[new_nodes_mask])

        merged[ntype] = torch.cat(merged[ntype], dim=0)
    return merged


def get_edge_index(nx_graph: nx.Graph, nodes_A: Union[List[str], np.array],
                   nodes_B: Union[List[str], np.array]) -> torch.LongTensor:
    biadj = nx.bipartite.biadjacency_matrix(nx_graph,
                                            row_order=nodes_A,
                                            column_order=nodes_B,
                                            format="coo")
    edge_index = torch.stack([torch.tensor(biadj.row, dtype=torch.long),
                              torch.tensor(biadj.col, dtype=torch.long)])
    return edge_index
