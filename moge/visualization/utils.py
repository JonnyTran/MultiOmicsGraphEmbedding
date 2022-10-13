from plotly import graph_objects as go

colors = ["aliceblue", "aqua", "aquamarine", "azure", "beige", "bisque", "black", "blanchedalmond",
          "blue", "blueviolet", "brown", "burlywood", "cadetblue", "chartreuse", "chocolate", "coral", "cornflowerblue",
          "cornsilk", "crimson", "cyan", "darkblue", "darkcyan", "darkgoldenrod", "darkgray", "darkgrey", "darkgreen",
          "darkkhaki", "darkmagenta", "darkolivegreen", "darkorange", "darkorchid", "darkred", "darksalmon",
          "darkseagreen", "darkslateblue", "darkslategray", "darkslategrey", "darkturquoise", "darkviolet", "deeppink",
          "deepskyblue", "dimgray", "dimgrey", "dodgerblue", "firebrick", "forestgreen", "fuchsia",
          "gainsboro", "gold", "goldenrod", "gray", "grey", "green", "greenyellow", "honeydew", "hotpink",
          "indianred", "indigo", "ivory", "khaki", "lavender", "lavenderblush", "lawngreen", "lemonchiffon",
          "lightblue", "lightcoral", "lightcyan", "lightgoldenrodyellow", "lightgray", "lightgrey", "lightgreen",
          "lightpink", "lightsalmon", "lightseagreen", "lightskyblue", "lightslategray", "lightslategrey",
          "lightsteelblue", "lightyellow", "lime", "limegreen", "linen", "magenta", "maroon", "mediumaquamarine",
          "mediumblue", "mediumorchid", "mediumpurple", "mediumseagreen", "mediumslateblue", "mediumspringgreen",
          "mediumturquoise", "mediumvioletred", "midnightblue", "mintcream", "mistyrose", "moccasin",
          "navy", "oldlace", "olive", "olivedrab", "orange", "orangered", "orchid", "palegoldenrod", "palegreen",
          "paleturquoise", "palevioletred", "papayawhip", "peachpuff", "peru", "pink", "plum", "powderblue", "purple",
          "red", "rosybrown", "royalblue", "rebeccapurple", "saddlebrown", "salmon", "sandybrown", "seagreen",
          "seashell", "sienna", "silver", "skyblue", "slateblue", "slategray", "slategrey", "snow", "springgreen",
          "steelblue", "tan", "teal", "thistle", "tomato", "turquoise", "violet", "wheat",
          "yellow", "yellowgreen"]

main_colors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'  # blue-teal
]


def configure_layout(fig, showlegend=True, showticklabels=False, showgrid=False, **kwargs) -> go.Figure:
    # Figure
    axis = dict(showline=False,  # hide axis line, grid, ticklabels and  title
                zeroline=False,
                showgrid=showgrid,
                showticklabels=showticklabels,
                title='',
                )

    fig.update_layout(
        **kwargs,
        showlegend=showlegend,
        # legend=dict(autosize=True, width=100),
        legend=dict(
            x=0,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        legend_orientation="v",
        autosize=True,
        margin=dict(
            l=5,
            r=5,
            b=5,
            t=30 if 'title' in kwargs else 5,
            pad=5
        ),
        xaxis=axis,
        yaxis=axis
    )
    return fig


def compress_legend(fig):
    group1_base, group2_base = fig.data[0].name.split(",")
    lines_marker_name = []
    for i, trace in enumerate(fig.data):
        part1, part2 = trace.name.split(',')
        if part1 == group1_base:
            lines_marker_name.append(
                {"line": trace.line.to_plotly_json(), "marker": trace.marker.to_plotly_json(), "mode": trace.mode,
                 "name": part2.lstrip(" ")})
        if part2 != group2_base:
            trace['name'] = ''
            trace['showlegend'] = False
        else:
            trace['name'] = part1

    ## Add the line/markers for the 2nd group
    for lmn in lines_marker_name:
        lmn["line"]["color"] = "black"
        lmn["marker"]["color"] = "black"
        fig.add_trace(go.Scatter(y=[None], **lmn))

    return fig.update_layout(legend_title_text='',
                             legend_itemclick=False,
                             legend_itemdoubleclick=False)
