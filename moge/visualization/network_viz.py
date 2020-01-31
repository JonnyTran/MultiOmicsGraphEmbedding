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
np.random.shuffle(color)

def hash_color(labels):
    sorted_labels = sorted(set(labels), reverse=True)
    colormap = {item: color[sorted_labels.index(item) % len(color)] for item in set(labels)}
    colors = [colormap[n] if n in colormap.keys() else None for n in labels]
    return colors

def graph_viz(g: nx.Graph,
              nodelist: list, node_symbol=None, node_color=None,
              edge_label: str = None, max_edges=50000,
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
    # fig = px.scatter(x=node_x, y=node_y,
    #                  hover_name=nodelist,
    #                  # symbol=node_symbol if node_symbol is not None else None,
    #                  # color=node_color if node_color is not None else None,
    #                  )

    # Edges data
    edge_data = pd.DataFrame([{"x": [pos[edge[0]][0], pos[edge[1]][0], None],
                               "y": [pos[edge[0]][1], pos[edge[1]][1], None],
                               **edge[2]} for edge in g.subgraph(nodelist).edges(data=True)])
    # Samples only certain edges
    if edge_data.shape[0] > max_edges:
        edge_data = edge_data.sample(n=max_edges)

    print("nodes", len(node_x), "edge_data", edge_data.shape[0], edge_data.columns.tolist())
    fig = px.line(edge_data, x="x", y="y",
                  color=edge_label,
                  )
    fig.add_scatter(x=node_x, y=node_y,
                    hover_name=nodelist,
                    # symbol=node_symbol if node_symbol is not None else None,
                    # color=node_color if node_color is not None else None,
                    )

    # hoverinfo='none')
    # fig.add_scatter(x=edge_data["x"], y=edge_data["y"],
    #                 mode='lines',
    #                 line=dict(
    #                     # color=hash_color(edge_data[edge_label]) if edge_label else 'rgb(210,210,210)',
    #                     color='rgb(50,50,50)',
    #                     width=1,
    #                 ),
    #                 showlegend=True,
    #                 # hoverinfo='none'
    #                 )

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
        xaxis=axis,
        yaxis=axis
    )

    return fig
