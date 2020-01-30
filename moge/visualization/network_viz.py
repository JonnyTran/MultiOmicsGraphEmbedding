import networkx as nx
import pandas as pd
import plotly.express as px
from fa2 import ForceAtlas2

forceatlas2 = ForceAtlas2(
    # Behavior alternatives
    outboundAttractionDistribution=True,  # Dissuade hubs
    linLogMode=False,  # NOT IMPLEMENTED
    adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
    edgeWeightInfluence=1.0,

    # Performance
    jitterTolerance=1.0,  # Tolerance
    barnesHutOptimize=True,
    barnesHutTheta=1.2,
    multiThreaded=False,  # NOT IMPLEMENTED

    # Tuning
    scalingRatio=2.0,
    strongGravityMode=False,
    gravity=1.0,

    # Log
    verbose=False)


def graph_viz(g: nx.Graph,
              nodelist: list, node_symbol=None, node_color=None,
              edge_label=None,
              title=None,
              pos=None, iterations=100):
    if pos is None:
        pos = forceatlas2.forceatlas2_networkx_layout(g.subgraph(nodelist), pos=None, iterations=iterations)
    if node_symbol is not None and node_symbol.isna().any():
        node_symbol.fillna("nan", inplace=True)

    node_x, node_y = zip(*[(pos[node][0], pos[node][1])
                           for node in nodelist])
    edge_data = pd.DataFrame([{"x": [pos[edge[0]][0], pos[edge[1]][0], None],
                               "y": [pos[edge[0]][1], pos[edge[1]][1], None],
                               **edge[2]  # edge d
                               }
                              for edge in g.subgraph(nodelist).edges(data=True)])

    fig = px.line(edge_data, x="x", y="y",
                  width=1,
                  color=edge_label if edge_label else 'rgb(210,210,210)',
                  )

    fig.add_scatter(x=node_x, y=node_y,
                    hover_name=nodelist,
                    symbol=node_symbol if node_symbol is not None else None,
                    color=node_color if node_color is not None else None,
                    title=title)

    return fig
