import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix

import seaborn

def matrix_heatmap(data):
    # Scatter plot of the graph adjacency matrix
    fig, ax = plt.subplots(figsize=(15, 15))

    data = np.ma.masked_invalid(data)

    heatmap = ax.pcolormesh(data, cmap=plt.cm.seismic,
                        vmin=np.nanmin(data), vmax=np.nanmax(data))
    # https://stackoverflow.com/a/16125413/190597 (Joe Kington)
    ax.patch.set(hatch='x', edgecolor='black')
    fig.colorbar(heatmap)

    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    # ax.set_xticklabels(row_labels, minor=False)
    # ax.set_yticklabels(column_labels, minor=False)
    plt.show()

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

