import torch

import pandas as pd


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
