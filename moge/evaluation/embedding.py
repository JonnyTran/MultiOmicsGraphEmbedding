import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr

from ..evaluation.link_prediction import largest_indices
from ..network.semantic_similarity import pairwise_distances, gower_distance


def distances_correlation(embeddings, sequences: pd.DataFrame, index: pd.Index, n_nodes=200):
    embedding_cov = pairwise_distances(embeddings, metric="euclidean", n_jobs=-2)
    assert embeddings.shape[0] == index.shape[0]

    top_k_farthest = index[np.concatenate(largest_indices(embedding_cov, k=int(n_nodes / 4), smallest=False))]
    top_k_closest = index[np.concatenate(largest_indices(embedding_cov, k=int(n_nodes / 4), smallest=True))]
    nodelist = top_k_farthest.append(top_k_closest)

    embedding_distances = pairwise_distances(pd.DataFrame(embeddings, index=index).loc[nodelist],
                                             metric="euclidean", n_jobs=-2)
    embedding_distances = squareform(embedding_distances, checks=False)

    seq_distances = gower_distance(sequences.loc[nodelist], )
    print(f"embedding dists {embedding_distances.shape}, seq dists {seq_distances.shape}")
    return pearsonr(x=embedding_distances, y=seq_distances)
