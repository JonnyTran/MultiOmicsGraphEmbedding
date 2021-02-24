import numpy as np
import pandas as pd
import torch


def edge_sizes(edge_index_dict):
    return {k: v.shape[1] for k, v in edge_index_dict.items()}


def edge_dict_intersection(edge_index_dict_A, edge_index_dict_B):
    inters = {}
    for metapath, edge_index in edge_index_dict_A.items():
        if metapath not in edge_index_dict_B:
            inters[metapath] = 0
            continue

        A = pd.DataFrame(edge_index.T.numpy(), columns=["source", "target"])
        B = pd.DataFrame(edge_index_dict_B[metapath].T.numpy(), columns=["source", "target"])
        int_df = pd.merge(A, B, how='inner', on=["source", "target"], sort=True)
        int_df = int_df[~int_df.duplicated()]
        inters[metapath] = torch.tensor(int_df.to_numpy().T, dtype=torch.long)

    return inters


def nonduplicate_indices(edge_index):
    edge_df = pd.DataFrame(edge_index.t().numpy())  # shape: (n_edges, 2)
    return ~edge_df.duplicated(subset=[0, 1])


def merge_node_index(old_node_index, new_node_index):
    merged = {}
    for ntype, new_nodes in new_node_index.items():
        if ntype not in old_node_index:
            merged.setdefault(ntype, []).append(new_nodes)
        else:
            merged.setdefault(ntype, []).append(old_node_index[ntype])
            new_nodes_mask = np.isin(new_nodes, old_node_index[ntype], invert=True)
            merged[ntype].append(new_nodes[new_nodes_mask])
        merged[ntype] = torch.cat(merged[ntype], 0)
    return merged
