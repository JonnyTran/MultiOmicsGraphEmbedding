import logging
from typing import Dict, List, Any, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pandas import DataFrame, Series

from moge.visualization.utils import configure_layout


def node_degree_viz(node_degrees, x_label, y_label, width=500, height=90):
    fig = go.Figure(data=go.Heatmap(z=node_degrees.applymap(lambda x: np.log10(x + 1)),
                                    x=node_degrees.columns, y=node_degrees.index, colorscale="Greys"),
                    layout=go.Layout(
                        xaxis=dict(title=x_label),
                        yaxis=dict(title=y_label),
                        width=width,
                        height=height,
                        margin=dict(l=5, r=5, b=5, t=5, pad=5),
                        font=dict(size=12, ),
                    ))
    return fig


def force_layout(g: nx.Graph, nodelist: List[str], iterations=100, init_pos=None) -> Dict[Any, Tuple[float, float]]:
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
        multiThreaded=True,
        # Tuning
        scalingRatio=2.0,
        strongGravityMode=True,
        gravity=1.0,
        # Log
        verbose=False)

    pos = forceatlas2.forceatlas2_networkx_layout(g.subgraph(nodelist), pos=init_pos, iterations=iterations)

    return pos


def graph_viz(g: nx.Graph,
              nodelist: List[str],
              node_symbol: pd.Series = None,
              node_color: pd.Series = None,
              node_size: pd.Series = None,
              edge_label: str = None,
              max_edges: int = 10000,
              title: str = None, width=1000, height=800, showlegend: bool = True,
              pos: Dict[str, np.ndarray] = None, iterations=100,
              **kwargs) -> go.Figure:
    if pos is None:
        pos = force_layout(g, nodelist, iterations)
    else:
        pos = process_pos(pos)

    # Nodes data
    node_symbol = process_labels(node_symbol)
    node_color = process_labels(node_color)
    node_size = process_labels(node_size)

    node_x, node_y = zip(*[(pos[node][0], pos[node][1])
                           for node in nodelist])

    if isinstance(node_color, pd.Series) and node_color.dtype == str and node_color.str.contains("#").any():
        express_mode = False
    else:
        express_mode = True

    if express_mode:
        logging.info("express_mode")

        if node_symbol is not None:
            node_symbol = match_labels(node_symbol, nodelist=nodelist)
        if node_color is not None:
            node_color = match_labels(node_color, nodelist=nodelist)

        fig = px.scatter(x=node_x, y=node_y,
                         hover_name=nodelist,
                         symbol=node_symbol if node_symbol is not None else None,
                         color=node_color if node_color is not None else None,
                         size=node_size if node_size is not None else None,
                         **kwargs)
    else:
        logging.info("Not express_mode")

        fig = go.Figure()
        fig.add_scatter3d(x=node_x, y=node_y,
                          mode='markers',
                          text=nodelist,
                          marker=dict(color=node_color,
                                      size=5,
                                      ),
                          **kwargs)

    # Edges data
    edges = list(g.subgraph(nodelist).edges(data=True if edge_label else False))
    # Samples only certain edges
    if max_edges and len(edges) > max_edges:
        np.random.shuffle(edges)
        edges = edges[:max_edges]

    if edge_label:
        plot_edge_w_labels(fig, edges=edges, edge_label=edge_label, pos=pos, plot3d=False)
    else:
        plot_edges(fig, edges=edges, pos=pos, plot3d=False)

    configure_layout(
        fig, showlegend=showlegend, height=height, width=width, title=title,
    ).update_traces(
        marker=dict(line=dict(width=0)),
        selector=dict(mode='markers'),
        marker_sizemin=4,
    ).update_layout(
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_zeroline=False,
        yaxis_zeroline=False,
        legend=dict(
            x=0,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        barmode='group',
        # paper_bgcolor='rgba(255,255,255,0)',
        # plot_bgcolor='rgba(255,255,255,0)',
    )

    return fig


def graph_viz3d(g: nx.Graph,
                nodelist: List[str], node_symbol=None, node_color=None, nodeline_color=None, node_text=None,
                edge_label: str = None, max_edges=10000,
                title=None, width=1000, height=800,
                pos: Union[DataFrame, Dict[str, Dict[str, float]]] = None,
                showlegend=True, **kwargs):
    if pos is None:
        raise Exception("Must provide pos as dict, i.e. {<node>:<3d coordinates>}")
    else:
        pos = process_pos(pos)

    # Nodes data
    node_symbol = process_labels(node_symbol)
    node_color = process_labels(node_color)

    node_x, node_y, node_z = zip(*[(pos[node][0], pos[node][1], pos[node][2])
                                   for node in nodelist])

    # Select express mode only if node_color is categorical strings
    if isinstance(node_color, pd.Series) and node_color.dtype == str and node_color.str.contains("#").any():
        # if node_color contains hex colors (e.g. #aaaaa)
        express_mode = False
    elif isinstance(node_color, pd.Series) and node_color.map(lambda x: isinstance(x, tuple) and len(x) == 3).any():
        # if node_color are a tuple of RGB values
        express_mode = False
        node_color = node_color.map(lambda rgb: [val / (max(rgb)) for val in rgb])  # change values
        node_color = node_color.map(
            lambda rgb: f"rgba({int(255 * rgb[0])}, {int(255 * rgb[1])}, {int(255 * rgb[2])}, 1.0)")
    else:
        express_mode = True

    if express_mode:
        print("express_mode")

        if node_symbol is not None:
            node_symbol = match_labels(node_symbol, nodelist=nodelist)
        if node_color is not None:
            node_color = match_labels(node_color, nodelist=nodelist)

        fig = px.scatter_3d(x=node_x, y=node_y, z=node_z, size_max=5,
                            hover_name=nodelist,
                            symbol=node_symbol if node_symbol is not None else None,
                            color=node_color if node_color is not None else None,
                            # text=node_text,
                            color_continuous_scale='HSV',
                            **kwargs)
    else:
        fig = go.Figure()
        fig.add_scatter3d(x=node_x, y=node_y, z=node_z,
                          mode='markers',
                          text=node_text,
                          marker=dict(color=node_color,
                                      size=5,
                                      line=dict(color=nodeline_color, width=50) if nodeline_color is not None else None,
                                      ),
                          **kwargs)

    # Edges data
    edges = list(g.subgraph(nodelist).edges(data=True if edge_label else False))

    # Samples only certain edges
    if max_edges and len(edges) > max_edges:
        np.random.shuffle(edges)
        edges = edges[:max_edges]
    print("nodes", len(node_x), "edges", len(edges))

    if edge_label:
        plot_edge_w_labels(fig, edges, edge_label, pos, plot3d=True)
    else:
        plot_edges(fig, edges, pos, plot3d=True)

    configure_layout(fig, showlegend=showlegend, height=height, width=width, title=title, )

    return fig


def process_pos(input: Union[DataFrame, Dict[str, Dict[str, float]]]) -> Dict[str, Tuple[float, float, float]]:
    if isinstance(input, DataFrame):
        cols = ["pos1", "pos2", "pos3"] if "pos3" in input.columns else ["pos1", "pos2"]
        pos = input[cols].to_dict(orient="index")
        pos = {node: [pos_dict["pos1"], pos_dict["pos2"], pos_dict["pos3"] if "pos3" in pos_dict else None] \
               for node, pos_dict in pos.items()}

    elif isinstance(input, dict) and isinstance(input[list(input.keys())[0]], dict):
        pos = {node: [pos_dict["pos1"], pos_dict["pos2"], pos_dict["pos3"] if "pos3" in pos_dict else None] \
               for node, pos_dict in input.items()}
    elif isinstance(input, dict) and isinstance(input[list(input.keys())[0]], list):
        pos = input

    else:
        raise Exception(f"Input is not of right type: {input}")

    return pos


def plot_edges(fig: go.Figure, edges: List[Tuple[str, str]], pos: Dict[str, Tuple[float, float, float]], plot3d=True):
    Xed, Yed, Zed = [], [], []
    for edge in edges:
        source, target = edge[0], edge[1]
        # if source not in pos or target not in pos: continue
        Xed += [pos[source][0], pos[target][0], None]
        Yed += [pos[source][1], pos[target][1], None]
        if plot3d:
            Zed += [pos[source][2], pos[target][2], None]

    if plot3d:
        fig.add_scatter3d(x=Xed, y=Yed, z=Zed,
                          mode='lines',
                          name='edges, ' + str(len(Xed)),
                          line=dict(
                              # color=hash_color(edge_data[edge_label]) if edge_label else 'rgb(210,210,210)',
                              color='rgb(50,50,50)',
                              width=0.25, ),
                          # showlegend=True,
                          hoverinfo='none')
    else:
        fig.add_scatter(x=Xed, y=Yed,
                        mode='lines',
                        name='edges, ' + str(len(Xed)),
                        line=dict(
                            # color=hash_color(edge_data[edge_label]) if edge_label else 'rgb(210,210,210)',
                            color='rgb(50,50,50)',
                            width=0.25, ),
                        # showlegend=True,
                        hoverinfo='none')


def plot_edge_w_labels(fig: go.Figure, edges: List[Tuple[str, str, str]], edge_label: str,
                       pos: Dict[str, Tuple[float, float, float]], plot3d=True):
    Xed_by_label, Yed_by_label, Zed_by_label = {}, {}, {}
    for edge in edges:
        label = edge[2][edge_label]
        Xed_by_label.setdefault(label, []).extend([pos[edge[0]][0], pos[edge[1]][0], None])
        Yed_by_label.setdefault(label, []).extend([pos[edge[0]][1], pos[edge[1]][1], None])
        if plot3d: Zed_by_label.setdefault(label, []).extend([pos[edge[0]][2], pos[edge[1]][2], None])

    if plot3d:
        for label in Xed_by_label:
            fig.add_scatter3d(x=Xed_by_label[label], y=Yed_by_label[label], z=Zed_by_label[label],
                              mode='lines',
                              name=label + ", size:" + str(len(edges)),
                              line=dict(
                                  color=label,
                                  colorscale="Viridis",
                                  width=0.5, ),
                              # showlegend=True,
                              hoverinfo='none')
    else:
        for label in Xed_by_label:
            fig.add_scatter(x=Xed_by_label[label], y=Yed_by_label[label],
                            mode='lines',
                            name=label + ", " + str(len(Xed_by_label[label])),
                            line=dict(
                                # color=hash_color([label])[0],
                                width=0.5, ),
                            # showlegend=True,
                            hoverinfo='none')


def process_labels(labels: pd.Series, delim="|"):
    if labels is not None and type(labels) is pd.Series:
        if labels.isna().any():
            labels.fillna("None", inplace=True)
        if labels.dtype == "object" and labels.str.contains(delim).any():
            labels = labels.str.split(delim, expand=True)[0].astype(str)
    return labels


def match_labels(labels: Union[Series, Dict[str, str]], nodelist: List[str], null_val=-1):
    # If nodelist is passed. select labels for nodes in nodelist, fill null_values if necessary
    if nodelist is not None:
        labels = nodelist.map(lambda x: labels.get(x, null_val))
        assert labels.shape == nodelist.shape

    return labels
