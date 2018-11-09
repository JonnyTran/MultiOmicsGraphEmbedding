import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE

import random


def visualize_embedding(embedding, network_train, edgelist=None, top_k=10000,
                        node_label="locus_type", cmap="viridis"):
    nodelist = embedding.node_list
    node_pos = embedding.get_tsne_node_pos()

    if edgelist is None:
        edgelist = embedding.get_top_k_predicted_edges(edge_type="d", top_k=top_k,
                                                       node_list=nodelist, training_network=network_train)

    if node_label is not None:
        genes_info = network_train.genes_info
        # node_with_lable = genes_info[node_label].notnull().index
        # nodelist = [node for node in nodelist if node in node_with_lable]
        node_labels = genes_info.loc[nodelist][node_label].astype(str)
        sorted_node_labels = sorted(node_labels.unique(), reverse=random.choice([True, False]))
        node_colormap = {f: sorted_node_labels.index(f) / len(sorted_node_labels) for f in node_labels.unique()}
        node_colors = [node_colormap[n] if n in node_colormap.keys() else None for n in node_labels]

        plot_embedding2D(node_pos, node_list=embedding.node_list, node_colors=node_colors,
                         legend=True, node_labels=node_labels, node_colormap=node_colormap, legend_size=20,
                         di_graph=network_train.G, cmap=cmap, nodelist=nodelist,
                         plot_nodes_only=False, edgelist=edgelist,
                         with_labels=False, figsize=(20, 15))
    else:
        plot_embedding2D(node_pos, node_list=embedding.node_list,
                         di_graph=network_train.G, cmap=cmap, nodelist=nodelist,
                         plot_nodes_only=False, edgelist=edgelist,
                         with_labels=False, figsize=(20, 15))


def plot_embedding2D(node_pos, node_list, di_graph=None,
                     legend=True, node_labels=None, node_colormap=None, legend_size=10,
                     node_colors=None, plot_nodes_only=True,
                     cmap="jet", file_name=None, figsize=(17, 15), **kwargs):
    node_num, embedding_dimension = node_pos.shape
    assert node_num == len(node_list)
    if(embedding_dimension > 2):
        print("Embedding dimension greater than 2, use tSNE to reduce it to 2")
        model = TSNE(n_components=2)
        node_pos = model.fit_transform(node_pos)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    if legend and node_labels is not None and node_colormap is not None and node_colors is not None:
        scalarMap = cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0, clip=True), cmap=cmap)
        top_node_labels = node_labels.value_counts()[:legend_size].index  # Get top k most popular legends labels
        for label in top_node_labels:
            ax.plot([0], [0],
                    color=scalarMap.to_rgba(node_colormap[label]),
                    label=label, linewidth=4) if label in node_colormap.keys() else None

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
                                   width=0.1, node_size=35,
                                   alpha=0.8, font_size=5, **kwargs)
        else:
            nx.draw_networkx(di_graph, pos,
                             node_color=node_colors, cmap=cmap, ax=ax,
                             width=0.1, node_size=25, arrows=True,
                             alpha=0.8, font_size=5, **kwargs)

        if legend:
            plt.legend(loc='best')
        plt.axis('off')

    if file_name:
        plt.savefig('%s_vis.pdf' % (file_name), dpi=300, format='pdf', bbox_inches='tight')
        plt.figure()


def get_node_color(node_labels):
    colors = [float(hash(s) % 256) / 256 for s in node_labels]

    return colors

