import warnings

import plotly.graph_objects as go
from matplotlib.colors import to_rgb
from pandas import DataFrame

from moge.visualization.utils import configure_layout


def plot_sankey_flow(nodes: DataFrame, links: DataFrame, opacity=0.6, font_size=8, orientation="h",
                     **kwargs):
    # change hex to rgba color to add opacity
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
        max_level = nodes['level'].max() - 1
        for layer in nodes['layer'].unique():
            level = (max_level - (nodes.query(f'layer == {layer}')['level'] - 1)).max()
            # if level != max_level:
            #     level += 1
            fig.add_vline(x=level / max_level, annotation_text=f'Layer {layer + 1}', layer='below',
                          line_dash="dot", line_color="gray", opacity=0.25, annotation_position="top left")

    fig = configure_layout(fig, paper_bgcolor='rgba(255,255,255,255)',
                           plot_bgcolor='rgba(0,0,0,0)', **kwargs)
    fig.update_layout(font_size=font_size)
    return fig
