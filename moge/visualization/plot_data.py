import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix


def matrix_heatmap(matrix, figsize=(12,12), cmap='gray', **kwargs):
    # Scatter plot of the graph adjacency matrix

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    if np.isnan(matrix).any():
        matrix = np.nan_to_num(matrix)

    cax = ax.matshow(matrix, interpolation='nearest', cmap=cmap, **kwargs)
    fig.colorbar(cax)

def plot_coo_matrix(m):
    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    ax.plot(m.col, m.row, 's', ms=1)
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

