import warnings

import plotly.graph_objects as go
from matplotlib.colors import to_rgb
from moge.visualization.utils import configure_layout
from pandas import DataFrame


def plot_sankey_flow(nodes: DataFrame, links: DataFrame, opacity=0.6, font_size=8, orientation="h",
                     **kwargs):
    # change '#fffff' to its 'rgba' value to add opacity
    rgba_colors = [f"rgba{tuple(int(val * 255) for val in to_rgb(color)) + (opacity if src != dst else 0,)}" \
                   for i, (src, dst, color) in links[['source', 'target', 'color']].iterrows()]

    if (nodes.index != nodes.reset_index().index).all():
        warnings.warn("`nodes.index` is not contiguous integer values.")

    fig = go.Figure(data=[
        go.Sankey(
            valueformat=".2f",
            orientation=orientation,
            arrangement="snap",
            # Define nodes
            node=dict(
                pad=5,
                thickness=15,
                # line=dict(color="black", width=np.where(nodes['level'] % 2, 0, 0)),
                label=nodes['label'],
                color=nodes['color'],
                customdata=nodes['count'],
                hovertemplate='num_nodes: %{customdata}',
            ),
            # Add links between nodes
            link=dict(
                label=links['label'],
                source=links['source'],
                target=links['target'],
                value=links['mean'],
                color=rgba_colors,
                # hoverlabel=dict(align='left'),
                customdata=links['std'],
                hovertemplate='%{label}: %{value} ± %{customdata:.3f}',
            ))],
        layout_xaxis_range=[0, 1],
        layout_yaxis_range=[0, 1],
    )

    if 'layer' in nodes.columns:
        n_layers = nodes['layer'].nunique()
        for layer in reversed(range(1, n_layers + 1)):
            fig.add_vline(x=layer / n_layers, annotation_text=f'Layer {layer}', layer='below',
                          line_dash="dash", line_color="gray", opacity=0.25, annotation_position="bottom left")

    fig = configure_layout(fig, paper_bgcolor='rgba(255,255,255,255)',
                           plot_bgcolor='rgba(0,0,0,0)', **kwargs)
    fig.update_layout(font_size=font_size)
    return fig
