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
    multiThreaded=False,
    # Tuning
    scalingRatio=2.0,
    strongGravityMode=False,
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


def hash_color(labels):
    sorted_node_labels = sorted(set(labels), reverse=True)
    colormap = {node: color[sorted_node_labels.index(node) % len(color)] for node in set(labels)}
    colors = [colormap[n] if n in colormap.keys() else None for n in labels]
    return colors


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

    fig = px.scatter(x=node_x, y=node_y,
                     hover_name=nodelist,
                     symbol=node_symbol if node_symbol is not None else None,
                     color=node_color if node_color is not None else None,
                     title=title)

    fig.add_trace(px.line(edge_data, x="x", y="y",
                          color=edge_label if edge_label else 'rgb(210,210,210)'))

    # fig.add_scatter(x=edge_data["x"], y=edge_data["y"],
    #                 mode='lines',
    #                 line=dict(width=1),
    #                 fillcolor=hash_color(edge_data[edge_label]) if edge_label else 'rgb(210,210,210)',
    #                 hoverinfo='none')

    return fig
