import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix

import seaborn


def matrix_heatmap(matrix, figsize=(15, 15)):
    # Scatter plot of the graph adjacency matrix

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix, interpolation='nearest', cmap=plt.cm.gray)
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

