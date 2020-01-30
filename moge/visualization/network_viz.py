import chart_studio.plotly as py
import networkx as nx
import plotly.graph_objects as go
from igraph.layout import Layout


def graph_viz(g: nx.Graph, nodelist, title="Graph"):
    pos = nx.spring_layout(g.subgraph(nodelist))
    Xv = [pos[k][0] for k in nodelist]
    Yv = [pos[k][1] for k in nodelist]

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

    layout = Layout(title=title,
                    font=dict(size=12),
                    showlegend=False,
                    autosize=True,
                    hovermode='closest',
                    annotations=[
                        dict(
                            showarrow=False,
                            text='This igraph.Graph has the Kamada-Kawai layout',
                            xref='paper',
                            yref='paper',
                            x=0,
                            y=-0.1,
                            xanchor='left',
                            yanchor='bottom',
                            font=dict(
                                size=14
                            )
                        )
                    ]
                    )

    data1 = [edges, nodes]
    fig1 = go.Figure(data=data1, layout=layout)
    return py.iplot(fig1)
