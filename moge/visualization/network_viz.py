import networkx as nx
import numpy as np
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
    multiThreaded=False,
    # Tuning
    scalingRatio=2.0,
    strongGravityMode=True,
    gravity=1.0,
    # Log
    verbose=False)

color = ["aliceblue", "antiquewhite", "aqua", "aquamarine", "azure",
         "beige", "bisque", "black", "blanchedalmond", "blue",
         "blueviolet", "brown", "burlywood", "cadetblue",
         "chartreuse", "chocolate", "coral", "cornflowerblue",
         "cornsilk", "crimson", "cyan", "darkblue", "darkcyan",
         "darkgoldenrod", "darkgray", "darkgrey", "darkgreen",
         "darkkhaki", "darkmagenta", "darkolivegreen", "darkorange",
         "darkorchid", "darkred", "darksalmon", "darkseagreen",
         "darkslateblue", "darkslategray", "darkslategrey",
         "darkturquoise", "darkviolet", "deeppink", "deepskyblue",
         "dimgray", "dimgrey", "dodgerblue", "firebrick",
          "floralwhite", "forestgreen", "fuchsia", "gainsboro",
          "ghostwhite", "gold", "goldenrod", "gray", "grey", "green",
          "greenyellow", "honeydew", "hotpink", "indianred", "indigo",
          "ivory", "khaki", "lavender", "lavenderblush", "lawngreen",
          "lemonchiffon", "lightblue", "lightcoral", "lightcyan",
          "lightgoldenrodyellow", "lightgray", "lightgrey",
          "lightgreen", "lightpink", "lightsalmon", "lightseagreen",
          "lightskyblue", "lightslategray", "lightslategrey",
          "lightsteelblue", "lightyellow", "lime", "limegreen",
          "linen", "magenta", "maroon", "mediumaquamarine",
         "mediumblue", "mediumorchid", "mediumpurple",
         "mediumseagreen", "mediumslateblue", "mediumspringgreen",
         "mediumturquoise", "mediumvioletred", "midnightblue",
         "mintcream", "mistyrose", "moccasin", "navajowhite", "navy",
         "oldlace", "olive", "olivedrab", "orange", "orangered",
         "orchid", "palegoldenrod", "palegreen", "paleturquoise",
         "palevioletred", "papayawhip", "peachpuff", "peru", "pink",
         "plum", "powderblue", "purple", "red", "rosybrown",
         "royalblue", "rebeccapurple", "saddlebrown", "salmon",
         "sandybrown", "seagreen", "seashell", "sienna", "silver",
         "skyblue", "slateblue", "slategray", "slategrey", "snow",
         "springgreen", "steelblue", "tan", "teal", "thistle", "tomato",
         "turquoise", "violet", "wheat", "white", "whitesmoke",
         "yellow", "yellowgreen"]
np.random.shuffle(color)

def hash_color(labels):
    sorted_labels = sorted(set(labels), reverse=True)
    colormap = {item: color[sorted_labels.index(item) % len(color)] for item in set(labels)}
    colors = [colormap[n] if n in colormap.keys() else None for n in labels]
    return colors

def graph_viz(g: nx.Graph,
              nodelist: list, node_symbol=None, node_color=None,
              edge_label: str = None, max_edges=10000,
              title=None, width=1000, height=800,
              pos=None, iterations=100, ):
    if pos is None:
        pos = forceatlas2.forceatlas2_networkx_layout(g.subgraph(nodelist), pos=None, iterations=iterations)

    # Nodes data
    if node_symbol is not None and type(node_symbol) is pd.Series:
        if node_symbol.isna().any():
            node_symbol.fillna("None", inplace=True)
        if node_symbol.dtype == "object" and node_symbol.str.contains("|").any():
            node_symbol = node_symbol.str.split("|", expand=True)[0].astype(str)
    if node_color is not None and type(node_color) is pd.Series:
        if node_color.isna().any():
            node_color.fillna("None", inplace=True)
        if node_color.dtype == "object" and node_color.str.contains("|").any():
            node_color = node_color.str.split("|", expand=True)[0].astype(str)

    node_x, node_y = zip(*[(pos[node][0], pos[node][1])
                           for node in nodelist])
    fig = px.scatter(x=node_x, y=node_y,
                     hover_name=nodelist,
                     symbol=node_symbol if node_symbol is not None else None,
                     color=node_color if node_color is not None else None,
                     )

    # Edges data

    edges = list(g.subgraph(nodelist).edges(data=False))
    np.random.shuffle(edges)

    # Samples only certain edges
    if max_edges and len(edges) > max_edges:
        edges = edges[:max_edges]

    if edge_label:
        Xed_by_label = {}
        Yed_by_label = {}
        for edge in edges:
            label = edge[2][edge_label]
            Xed_by_label.setdefault(label, []).extend([pos[edge[0]][0], pos[edge[1]][0], None])
            Yed_by_label.setdefault(label, []).extend([pos[edge[0]][1], pos[edge[1]][1], None])

        for label in Xed_by_label:
            fig.add_scatter(x=Xed_by_label[label], y=Yed_by_label[label],
                            mode='lines',
                            name=label + ", " + str(len(Xed_by_label[label])),
                            line=dict(
                                color=hash_color([label])[0],
                                # color='rgb(50,50,50)',
                                width=0.5, ),
                            # showlegend=True,
                            hoverinfo='none')
    else:
        Xed, Yed = [], []
        for edge in edges:
            Xed += [pos[edge[0]][0], pos[edge[1]][0], None]
            Yed += [pos[edge[0]][1], pos[edge[1]][1], None]

        print("nodes", len(node_x), "edges", len(edges))
        fig.add_scatter(x=Xed, y=Yed,
                        mode='lines',
                        name='edges, ' + str(len(Xed)),
                        line=dict(
                            # color=hash_color(edge_data[edge_label]) if edge_label else 'rgb(210,210,210)',
                            color='rgb(50,50,50)',
                            width=0.25, ),
                        # showlegend=True,
                        hoverinfo='none')

    # Figure
    axis = dict(showline=False,  # hide axis line, grid, ticklabels and  title
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title=''
                )
    fig.update_layout(
        title=title,
        autosize=True,
        width=width,
        height=height,
        margin=dict(
            l=10,
            r=10,
            b=10,
            t=10,
            pad=4
        ),
        xaxis=axis,
        yaxis=axis
    )

    return fig


def graph_viz3d(g: nx.Graph,
                nodelist: list, node_symbol=None, node_color=None,
                edge_label: str = None, max_edges=10000,
                title=None, width=1000, height=800,
                pos=None, iterations=100, ):
    if pos is None:
        raise Exception("Must provide pos as dict, i.e. {<node>:<3d coordinates>}")

    # Nodes data
    if node_symbol is not None and type(node_symbol) is pd.Series:
        if node_symbol.isna().any():
            node_symbol.fillna("None", inplace=True)
        if node_symbol.dtype == "object" and node_symbol.str.contains("|").any():
            node_symbol = node_symbol.str.split("|", expand=True)[0].astype(str)
    if node_color is not None and type(node_color) is pd.Series:
        if node_color.isna().any():
            node_color.fillna("None", inplace=True)
        if node_color.dtype == "object" and node_color.str.contains("|").any():
            node_color = node_color.str.split("|", expand=True)[0].astype(str)

    node_x, node_y, node_z = zip(*[(pos[node][0], pos[node][1], pos[node][2])
                                   for node in nodelist])
    fig = px.scatter_3d(x=node_x, y=node_y, z=node_z,
                        hover_name=nodelist,
                        symbol=node_symbol if node_symbol is not None else None,
                        color=node_color if node_color is not None else None,
                        )

    # Edges data

    edges = list(g.subgraph(nodelist).edges(data=False))
    np.random.shuffle(edges)

    # Samples only certain edges
    if max_edges and len(edges) > max_edges:
        edges = edges[:max_edges]

    if edge_label:
        Xed_by_label, Yed_by_label, Zed_by_label = {}, {}, {}
        for edge in edges:
            label = edge[2][edge_label]
            Xed_by_label.setdefault(label, []).extend([pos[edge[0]][0], pos[edge[1]][0], None])
            Yed_by_label.setdefault(label, []).extend([pos[edge[0]][1], pos[edge[1]][1], None])
            Zed_by_label.setdefault(label, []).extend([pos[edge[0]][2], pos[edge[1]][3], None])

        for label in Xed_by_label:
            fig.add_scatter3d(x=Xed_by_label[label], y=Yed_by_label[label], z=Zed_by_label[label],
                              mode='lines',
                              name=label + ", " + str(len(Xed_by_label[label])),
                              line=dict(
                                  color=hash_color([label])[0],
                                  # color='rgb(50,50,50)',
                                  width=0.5, ),
                              # showlegend=True,
                              hoverinfo='none')
    else:
        Xed, Yed, Zed = [], [], []
        for edge in edges:
            Xed += [pos[edge[0]][0], pos[edge[1]][0], None]
            Yed += [pos[edge[0]][1], pos[edge[1]][1], None]
            Zed += [pos[edge[0]][2], pos[edge[1]][2], None]

        print("nodes", len(node_x), "edges", len(edges))
        fig.add_scatter3d(x=Xed, y=Yed, z=Zed,
                          mode='lines',
                          name='edges, ' + str(len(Xed)),
                          line=dict(
                              # color=hash_color(edge_data[edge_label]) if edge_label else 'rgb(210,210,210)',
                              color='rgb(50,50,50)',
                              width=0.25, ),
                          # showlegend=True,
                          hoverinfo='none')

    # Figure
    axis = dict(showline=False,  # hide axis line, grid, ticklabels and  title
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title=''
                )
    fig.update_layout(
        title=title,
        autosize=True,
        width=width,
        height=height,
        margin=dict(
            l=5,
            r=5,
            b=5,
            t=5,
            pad=5
        ),
        xaxis=axis,
        yaxis=axis
    )

    return fig
