import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE

def plot_embedding2D(node_pos, node_list, node_colors=None, di_graph=None):
    node_num, embedding_dimension = node_pos.shape
    assert node_num == len(node_list)
    if(embedding_dimension > 2):
        print("Embedding dimension greater than 2, use tSNE to reduce it to 2")
        model = TSNE(n_components=2)
        node_pos = model.fit_transform(node_pos)

    if di_graph is None:
        # plot using plt scatter
        plt.scatter(node_pos[:, 0], node_pos[:, 1], c=node_colors, cmap="jet")
    else:
        # plot using networkx with edge structure
        pos = {}
        for i in range(node_num):
            pos[i] = node_pos[i, :]
        if node_colors:
            nx.draw_networkx_nodes(di_graph, pos,
                                   node_color=node_colors, cmap="jet",
                                   width=0.1, node_size=100,
                                   arrows=False, alpha=0.8,
                                   font_size=5)
        else:
            nx.draw_networkx(di_graph, pos, node_color=node_colors,
                             width=0.1, node_size=300, arrows=False,
                             alpha=0.8, font_size=12)


def get_node_color(node_labels):
    colors = [float(hash(s) % 256) / 256 for s in node_labels]

    return colors

def expVis(X, res_pre, m_summ, node_labels=None, di_graph=None):
    print('\tGraph Visualization:')
    if node_labels:
        node_colors = get_node_color(node_labels)
    else:
        node_colors = None
    plot_embedding2D(X, node_colors=node_colors,
                     di_graph=di_graph)
    plt.savefig('%s_%s_vis.pdf' % (res_pre, m_summ), dpi=300,
                format='pdf', bbox_inches='tight')
    plt.figure()
