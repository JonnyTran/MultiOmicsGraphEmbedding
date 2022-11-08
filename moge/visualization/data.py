import datashader as ds
import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import xarray as xr
from datashader import reductions as rd
from holoviews.operation.datashader import rasterize
from plotly.subplots import make_subplots
from scipy.sparse import coo_matrix
from sklearn.metrics import classification_report

from moge.visualization.utils import configure_layout, group_axis_ticks

hv.extension('plotly')


def rasterize_matrix(df: pd.DataFrame, x_label="X", y_label="Y", size=1000, **kwargs):
    layout_args = dict()
    if isinstance(df, pd.DataFrame):
        y_label = df.index.name if df.index.name else df.index.names[0]
        x_label = df.columns.name if df.columns.name else df.columns.names[0]

        if isinstance(df.columns, pd.MultiIndex):
            grouped_xticks = group_axis_ticks(df.columns, level=0)
            layout_args.update(dict(
                xaxis_tickmode='array',
                xaxis_tickvals=grouped_xticks.values, xaxis_ticktext=grouped_xticks.index,
            ))

        if isinstance(df.index, pd.MultiIndex):
            grouped_yticks = group_axis_ticks(df.index, level=0)
            layout_args.update(dict(
                yaxis_tickmode='array',
                yaxis_tickvals=grouped_yticks.values, yaxis_ticktext=grouped_yticks.index,
            ))

    width = max(int(size * df.shape[1] / sum(df.shape)), 500)
    height = max(int(size * df.shape[0] / sum(df.shape)), 500)

    img = hv.Image((np.arange(df.shape[1]), np.arange(df.shape[0]), df))
    rasterized_img: hv.Image = rasterize(img, width=width, height=height)

    rasterized_img.opts(width=width, height=height, xlabel=x_label, ylabel=y_label)
    rasterized_img.opts(invert_yaxis=True, cmap=px.colors.sequential.Plasma, logz=True,
                        show_legend=True)

    fig = hv.render(rasterized_img, backend="plotly")

    layout_args.update(kwargs)
    go_fig = go.Figure(fig).update_layout(**layout_args)
    return go_fig


def heatmap_fast(df: pd.DataFrame, row_label="row", col_label="col", size=1000, agg="mean", **kwargs):
    if isinstance(df, pd.DataFrame):
        if isinstance(df.index, pd.MultiIndex):
            rows = df.index.get_level_values(0)
            row_label = df.index.names[0]
        else:
            rows, row_label = df.index, df.index.name

        if isinstance(df.columns, pd.MultiIndex):
            cols = df.columns.get_level_values(0)
            col_label = df.columns.names[0]
        else:
            cols, col_label = df.columns, df.columns.name

    else:
        rows = np.arange(df.shape[0])
        cols = np.arange(df.shape[1])

    pw_s = xr.DataArray(df, coords=[(row_label, rows), (col_label, cols)])

    plot_width = int(size * len(cols) / sum(df.shape))
    plot_height = int(size * len(rows) / sum(df.shape))
    cvs = ds.Canvas(plot_height=plot_height, plot_width=plot_width,
                    # x_range=(0, arr.shape[1]),
                    # y_range=(0, arr.shape[0])
                    )

    if agg == 'avg':
        agg_fn = rd.mean()
    elif agg == 'max':
        agg_fn = rd.max()
    elif agg == 'min':
        agg_fn = rd.min()
    else:
        agg_fn = rd.mean()

    agg = cvs.raster(pw_s, agg=agg_fn)

    if 'height' not in kwargs or 'width' not in kwargs:
        kwargs['height'] = plot_height
        kwargs['width'] = plot_width

    fig = px.imshow(agg, labels={"x": row_label, "y": col_label}) \
        .update_layout(autosize=True, **kwargs)
    return fig


def clf_report(y_true, y_pred, classes, threshold=0.5, top_k=20):
    results = pd.DataFrame(classification_report(y_true=y_true,
                                                 y_pred=(y_pred >= threshold),
                                                 target_names=classes,
                                                 output_dict=True)).T
    return results.sort_values(by="support", ascending=False)[:top_k]


def clf_report_compare(y_train, y_train_pred, y_test, y_test_pred, classes, threshold=0.5):
    train = clf_report(y_train, y_train_pred, classes, threshold=threshold)
    test = clf_report(y_test, y_test_pred, classes, threshold=threshold)

    train.columns = pd.MultiIndex.from_product([train.columns, ["train"]])
    test.columns = pd.MultiIndex.from_product([test.columns, ["test"]])
    return pd.concat([train, test], axis=1)



def heatmap_compare(y_true, y_pred, file_output=None, title=None, autosize=True, width=1400, height=700):
    if not hasattr(y_true, "columns"):
        columns = None
    elif type(y_true.columns) == pd.MultiIndex:
        columns = y_true.columns.to_series().apply(lambda x: '{0}-{1}'.format(*x))
    else:
        columns = y_true.columns

    fig = make_subplots(rows=1, cols=2, subplot_titles=("True Labels", "Predicted"))

    fig.add_trace(go.Heatmap(
        z=y_true,
        x=columns,
        y=y_true.index if hasattr(y_true, "index") else None,
        coloraxis="coloraxis1",
        hoverongaps=False),
        row=1, col=1)

    fig.add_trace(go.Heatmap(
        z=y_pred,
        x=columns,
        y=y_pred.index if hasattr(y_pred, "index") else None,
        coloraxis="coloraxis1",
        hoverongaps=False),
        row=1, col=2)

    fig = configure_layout(
        fig,
        title=title,
        width=width,
        height=height, ).update_layout(autosize=autosize, )

    if file_output:
        fig.write_image(file_output)

    return fig



def bar_chart(results: dict, measures, title=None, bar_width=0.08, loc="best"):
    methods = list(results.keys())
    y_pos = np.arange(len(measures))

    if type(measures) == str:
        performances = [results[method] for method in methods]

        plt.bar(y_pos, performances, align='center', alpha=0.5)
        plt.xticks(y_pos, methods)
        plt.ylabel(measures)

    elif type(measures) == list:
        n_groups = len(methods)
        performances = {}
        fig, ax = plt.subplots(dpi=300)
        index = np.arange(n_groups)

        color_dict = {"LINE": "b", "HOPE": "c", "SDNE": "y", "node2vec": "g", "BioVec": "m", "rna2rna": "r",
                      "siamese": "r",
                      "Databases": "k"}
        opacity = 0.8

        for method in methods:
            performances[method] = []
            for measure in measures:
                performances[method].append(results[method][measure])

        for idx, method in enumerate(methods):
            plt.bar(y_pos + idx * bar_width, performances[method], bar_width,
                    alpha=opacity,
                    color=color_dict[method],
                    label=method.replace("test_", ""))
        # plt.xlabel('Methods')
        plt.ylabel('Scores')
        plt.xticks(y_pos + bar_width * (n_groups / 2), measures)
        plt.legend(loc=loc)

    plt.tight_layout()
    plt.title(title)
    plt.show()

def plot_coo_matrix(m):
    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    ax.plot(m.col, m.row, 's', ms=1)
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    return ax
