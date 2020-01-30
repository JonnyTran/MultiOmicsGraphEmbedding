import networkx as nx
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


def graph_viz(g: nx.Graph, nodelist: list, annotations, labels=None, title="Graph", pos=None, iterations=100):
    if pos is None:
        pos = forceatlas2.forceatlas2_networkx_layout(g.subgraph(nodelist), pos=None, iterations=iterations)

    Xv = [pos[node][0] for node in nodelist]
    Yv = [pos[node][1] for node in nodelist]

    Xed = []
    Yed = []
    for edge in g.subgraph(nodelist).edges(data=False):
        Xed += [pos[edge[0]][0], pos[edge[1]][0], None]
        Yed += [pos[edge[0]][1], pos[edge[1]][1], None]

    # edge_trace = go.Scatter(x=Xed,
    #                         y=Yed,
    #                         mode='lines',
    #                         line=dict(color='rgb(210,210,210)', width=1),
    #                         hoverinfo='none'
    #                         )
    fig = px.scatter(x=Xv, y=Yv, color=labels if labels is not None else None, )
    fig.add_scatter(x=Xed, y=Yed, mode='lines', line=dict(color='rgb(210,210,210)', width=1), hoverinfo='none')
    # node_trace = go.Scatter(
    #     x=Xv, y=Yv,
    #     mode='markers',
    #     hoverinfo='text',
    #     marker=dict(
    #         showscale=True,
    #         # colorscale options
    #         # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
    #         # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
    #         # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
    #         colorscale='YlGnBu',
    #         reversescale=True,
    #         color=labels if labels is not None else None,
    #         size=10,
    #         colorbar=dict(
    #             thickness=15,
    #             title='Node Connections',
    #             xanchor='left',
    #             titleside='right'
    #         ),
    #         line_width=2))

    # fig1 = go.Figure(data=[edge_trace, node_trace])
    return fig
