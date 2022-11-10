import networkx as nx
import pandas as pd
from jaal import Jaal

from .plotly.graph import graph_viz, graph_viz3d
from .pyvis.graph import plot_graph

__all__ = ['graph_viz', 'graph_viz3d', 'plot_graph']


def jaal_graph_plot(G: nx.Graph):
    if isinstance(G, nx.MultiGraph):
        edges_df_names = ['from', 'to', 'group']
    else:
        edges_df_names = ['from', 'to']

    nodes = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index').reset_index(names=['id'])
    edges = pd.DataFrame.from_dict(dict(G.edges), orient='index').reset_index(names=edges_df_names)

    jaal = Jaal(edges, nodes)
    return jaal
