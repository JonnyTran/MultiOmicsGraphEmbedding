import plotly.graph_objects as go
from matplotlib.colors import to_rgb
from pandas import DataFrame

from moge.visualization.utils import configure_layout


def plot_sankey_flow(nodes: DataFrame, links: DataFrame, opacity=0.6, font_size=8, orientation="h",
                     **kwargs):
    # change '#fffff' to its 'rgba' value to add opacity
    rgba_colors = [f"rgba{tuple(int(val * 255) for val in to_rgb(color)) + (opacity,)}" \
                   for color in links['color']]

    fig = go.Figure(data=[go.Sankey(
        valueformat=".2f",
        orientation=orientation,
        arrangement="snap",
        # Define nodes
        node=dict(
            pad=5,
            thickness=15,
            line=dict(color="black", width=0.5),
            label=nodes['label'],
            color=nodes['color'],
            customdata=nodes['count'],
            hovertemplate='num_nodes: %{customdata}',
        ),
        # Add links
        link=dict(
            source=links['source'],
            target=links['target'],
            label=links['label'],
            value=links['mean'],
            color=rgba_colors,
            hovertemplate='attn weight: %{value} ± %{customdata:.3f}',
        ),

    )], )

    configure_layout(fig, **kwargs)
    fig.update_layout(font_size=font_size)
    return fig
