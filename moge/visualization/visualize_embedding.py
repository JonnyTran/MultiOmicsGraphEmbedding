import random

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE


def visualize_embedding(embedding, network, edgelist=None, top_k=5000, test_nodes=None,
                        node_label="locus_type", cmap="gist_ncar", **kwargs):
    nodelist = embedding.node_list
    node_pos = embedding.get_tsne_node_pos()

    if edgelist is None:
        edgelist = embedding.get_top_k_predicted_edges(edge_type="d", top_k=top_k,
                                                       node_list=nodelist, training_network=network)
        if len(edgelist[0]) > 2: # Has weight component
            edge_weights = [w for u,v,w in edgelist]
            edgelist = [(u, v) for u,v,w in edgelist]
            kwargs["edge_color"] = edge_weights
            kwargs["edge_cmap"] = cm.get_cmap("binary")
            kwargs["edge_vmin"] = 0.0
            kwargs["edge_vmax"] = 1.0
            kwargs["style"] = "dashed"

    if test_nodes is not None:
        labels_dict = {node:node for node in test_nodes if node in nodelist}
        kwargs["labels"] = labels_dict
        kwargs["with_labels"] = True
        kwargs["font_size"] = 2

    if node_label is not None:
        genes_info = network.genes_info
        node_labels = genes_info.loc[nodelist][node_label].str.split("|", expand=True)[0].astype(str)
        sorted_node_labels = sorted(node_labels.unique(), reverse=random.choice([True, False]))
        colors = np.linspace(0, 1, len(sorted_node_labels))
        node_colormap = {f: colors[sorted_node_labels.index(f)] for f in node_labels.unique()}
        node_colors = [node_colormap[n] if n in node_colormap.keys() else None for n in node_labels]

        plot_embedding2D(node_pos, node_list=embedding.node_list, node_colors=node_colors,
                         legend=True, node_labels=node_labels, node_colormap=node_colormap, legend_size=20,
                         di_graph=network.G, cmap=cmap, nodelist=nodelist,
                         plot_nodes_only=False, edgelist=edgelist,
                         figsize=(20, 15), **kwargs)
    else:
        plot_embedding2D(node_pos, node_list=embedding.node_list,
                         di_graph=network.G, cmap=cmap, nodelist=nodelist,
                         plot_nodes_only=False, edgelist=edgelist,
                         figsize=(20, 15), **kwargs)


def plot_embedding2D(node_pos, node_list, di_graph=None,
                     legend=True, node_labels=None, node_colormap=None, legend_size=10,
                     node_colors=None, plot_nodes_only=True,
                     cmap="viridis", file_name=None, figsize=(17, 15), **kwargs):
    node_num, embedding_dimension = node_pos.shape
    assert node_num == len(node_list)
    if(embedding_dimension > 2):
        print("Embedding dimension greater than 2, use tSNE to reduce it to 2")
        model = TSNE(n_components=2)
        node_pos = model.fit_transform(node_pos)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    if legend and node_labels is not None and node_colormap is not None and node_colors is not None:
        scalarMap = cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0, clip=False), cmap=cmap)
        top_node_labels = node_labels.value_counts()[:legend_size].index  # Get top k most popular legends labels
        for label in top_node_labels:
            ax.plot([0], [0],
                    color=scalarMap.to_rgba(node_colormap[label], norm=False),
                    label=label, linewidth=4) if label in node_colormap.keys() else None

    if "node_size" not in kwargs:
        kwargs["node_size"] = 25
    if "with_labels" not in kwargs or "labels" not in kwargs:
        kwargs["with_labels"] = False
    if "font_size" not in kwargs:
        kwargs["font_size"] = 5

    if di_graph is None:
        # Plot using plt scatter
        plt.scatter(node_pos[:, 0], node_pos[:, 1], c=node_colors, cmap=cmap)
    else:
        # Plot using networkx with edge structure
        pos = {}
        for i, node in enumerate(node_list):
            pos[node] = node_pos[i, :]

        if plot_nodes_only:
            nx.draw_networkx_nodes(di_graph, pos,
                                   node_color=node_colors, cmap=cmap, ax=ax,
                                   width=0.1,
                                   alpha=0.8, **kwargs)
        else:
            nx.draw_networkx(di_graph, pos,
                             node_color=node_colors, cmap=cmap, ax=ax,
                             width=0.1, arrows=True,
                             alpha=0.8, **kwargs)

        if legend:
            plt.legend(loc='best')
        plt.axis('off')

    if file_name:
        plt.savefig('%s_vis.pdf' % (file_name), dpi=300, format='pdf', bbox_inches='tight')
        plt.figure()


def get_node_color(node_labels):
    colors = [float(hash(s) % 256) / 256 for s in node_labels]

    return colors

