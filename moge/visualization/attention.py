import plotly.graph_objects as go

from moge.visualization.graph import configure_layout


def sankey_plot(nodes, links, **kwargs):
    # override gray link colors with 'source' colors
    opacity = 0.4
    # change 'magenta' to its 'rgba' value to add opacity

    fig = go.Figure(data=[go.Sankey(
        valueformat=".2f",
        # Define nodes
        node=dict(
            pad=15,
            thickness=15,
            line=dict(color="black", width=0.5),
            label=nodes['label'],
            color=nodes['color']
        ),
        # Add links
        link=dict(
            source=links['source'],
            target=links['target'],
            value=links['value'],
            label=links['label'],
            color=links['color'],
        ),

    )], )

    configure_layout(fig, **kwargs)
    fig.update_layout(font_size=10)
    return fig
