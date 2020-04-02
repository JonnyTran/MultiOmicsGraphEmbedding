import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.sparse import coo_matrix
from scipy.sparse import issparse
from sklearn.metrics import classification_report


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


def heatmap(table: pd.DataFrame, file_output=None, title=None, autosize=True, width=800, height=1000):
    if not hasattr(table, "columns"):
        columns = None
    elif type(table.columns) == pd.MultiIndex:
        columns = table.columns.to_series().apply(lambda x: '{0}-{1}'.format(*x))
    else:
        columns = table.columns

    fig = go.Figure(data=go.Heatmap(
        z=table,
        x=columns,
        y=table.index if hasattr(table, "index") else None,
        hoverongaps=False, ))

    fig.update_layout(
        title=title,
        autosize=autosize,
        width=width,
        height=height,
    )
    if file_output:
        fig.write_image(file_output)

    return fig

def heatmap_compare(y_true, y_pred, file_output=None, title=None, autosize=True, width=1400, height=700):
    if not hasattr(y_true, "columns"):
        columns = None
    elif type(y_true.columns) == pd.MultiIndex:
        columns = y_true.columns.to_series().apply(lambda x: '{0}-{1}'.format(*x))
    else:
        columns = y_true.columns

    fig = make_subplots(rows=1, cols=2)

    fig.append_trace(go.Heatmap(
        z=y_true,
        x=columns,
        y=y_true.index if hasattr(y_true, "index") else None,
        hoverongaps=False),
        row=1, col=1)

    fig.append_trace(go.Heatmap(
        z=y_pred,
        x=columns,
        y=y_pred.index if hasattr(y_pred, "index") else None,
        hoverongaps=False),
        row=1, col=2)

    fig.update_layout(
        title=title,
        autosize=autosize,
        width=width,
        height=height,
    )
    if file_output:
        fig.write_image(file_output)

    return fig


def plot_training_history(history, title=""):
    fig = go.Figure()
    for metric in history.history.keys():
        fig.add_trace(go.Scatter(x=np.arange(len(history.history["loss"])),
                                 y=history.history[metric], name=metric,
                                 mode='lines+markers'))
    fig.update_layout(
        title=title,
        xaxis_title="Iteration",
        yaxis_title="Percentage",
    )
    fig.show()


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

def matrix_heatmap(matrix, figsize=(12, 12), cmap='gray', **kwargs):
    # Scatter plot of the graph adjacency matrix

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    if issparse(matrix):
        matrix = matrix.todense()

    if np.isnan(matrix).any():
        matrix = np.nan_to_num(matrix)

    cax = ax.matshow(matrix, cmap=cmap, **kwargs)
    fig.colorbar(cax)

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
