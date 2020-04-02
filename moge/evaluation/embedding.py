import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr

from .utils import largest_indices
from ..network.semantic_similarity import pairwise_distances, gower_distance


def distances_correlation(embeddings, labels, index: pd.Index, n_nodes=200, verbose=False):
    embedding_cov = pairwise_distances(embeddings, metric="euclidean", n_jobs=-2)
    assert embeddings.shape[0] == index.shape[0]

    top_farthest_pairs = index[np.concatenate(largest_indices(embedding_cov, k=int(n_nodes / 4), smallest=False))]
    top_closest_pairs = index[np.concatenate(largest_indices(embedding_cov, k=int(n_nodes / 4), smallest=True))]
    nodelist = top_farthest_pairs.append(top_closest_pairs)

    embedding_distances = pairwise_distances(pd.DataFrame(embeddings, index=index).loc[nodelist],
                                             metric="euclidean", n_jobs=-2)
    embedding_distances = squareform(embedding_distances, checks=False)

    if isinstance(labels, pd.DataFrame):
        label_distances = gower_distance(labels.loc[nodelist], verbose=verbose)
    else:
        label_distances = pairwise_distances(pd.DataFrame(labels, index=index).loc[nodelist], metric="cosine",
                                             n_jobs=-2)
        label_distances = squareform(label_distances, checks=False)

    print(f"embedding dists {embedding_distances.shape}, seq dists {label_distances.shape}") if verbose else None

    r, p_val = pearsonr(x=embedding_distances[~np.isnan(label_distances)],
                        y=label_distances[~np.isnan(label_distances)])
    return r, p_val
