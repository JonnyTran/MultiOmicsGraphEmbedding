import networkx as nx
import plotly.graph_objects as go
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
    verbose=True)


def graph_viz(g: nx.Graph, nodelist: list, title="Graph"):
    pos = forceatlas2.forceatlas2_networkx_layout(g.subgraph(nodelist), pos=None, iterations=2000)
    print("spring_layout done")
    Xv = [pos[node][0] for node in nodelist]
    Yv = [pos[node][1] for node in nodelist]

    Xed = []
    Yed = []
    for edge in g.subgraph(nodelist).edges(data=False):
        Xed += [pos[edge[0]][0], pos[edge[1]][0], None]
        Yed += [pos[edge[0]][1], pos[edge[1]][1], None]

    edges = go.Scatter(x=Xed,
                       y=Yed,
                       mode='lines',
                       line=dict(color='rgb(210,210,210)', width=1),
                       hoverinfo='none'
                       )
    nodes = go.Scatter(x=Xv,
                       y=Yv,
                       mode='markers',
                       name='net',
                       marker=dict(symbol='circle-dot',
                                   size=5,
                                   color='#6959CD',
                                   line=dict(color='rgb(50,50,50)', width=0.5)
                                   ),
                       text=nodelist,
                       hoverinfo='text'
                       )

    data1 = [edges, nodes]
    fig1 = go.Figure(data=data1)
    return fig1
