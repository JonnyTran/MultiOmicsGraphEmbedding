from itertools import cycle

import numpy as np
import pandas as pd
from plotly import graph_objects as go
from sklearn.metrics import auc

from .utils import colors
from ..evaluation.classification import compute_roc_auc_curve


def plot_roc_curve(y_test, y_score, n_classes, sample_weight=None, width=700, height=700,
                   title='Receiver operating characteristic example'):
    # Compute ROC curve and ROC area for each class
    fpr, roc_auc, tpr = compute_roc_auc_curve(y_test, y_score, range(n_classes), sample_weight)

    trace1 = go.Scatter(x=fpr[2], y=tpr[2],
                        mode='lines',
                        line=dict(color='darkorange', width=2),
                        name='ROC curve (area = %0.2f)' % roc_auc[2])

    trace2 = go.Scatter(x=[0, 1], y=[0, 1],
                        mode='lines',
                        line=dict(color='navy', width=2, dash='dash'),
                        showlegend=False)

    fig = configure_figure([trace1, trace2], title, width, height)
    return fig


def plot_roc_curve_multiclass(y_test: pd.DataFrame, y_score, classes: (list, pd.Index), sample_weight=None,
                              title='ROC Curve (multi-class)',
                              width=800, height=700):
    if classes is not None and isinstance(y_test, pd.DataFrame):
        class_indices = y_test.columns.get_indexer(classes)
    else:
        class_indices = range(y_test.shape[1])

    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.values

    if isinstance(y_score, pd.DataFrame):
        y_score = y_score.values

    fpr, roc_auc, tpr = compute_roc_auc_curve(y_test, y_score, class_indices, sample_weight)
    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in class_indices]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in class_indices:
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(classes)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    data = []
    trace1 = go.Scatter(x=fpr["micro"], y=tpr["micro"],
                        mode='lines',
                        line=dict(color='deeppink', width=2, dash='dot'),
                        name='micro-average ROC curve (area={0:0.2f})'
                             ''.format(roc_auc["micro"]))
    data.append(trace1)

    trace2 = go.Scatter(x=fpr["macro"], y=tpr["macro"],
                        mode='lines',
                        line=dict(color='navy', width=2, dash='dot'),
                        name='macro-average ROC curve (area={0:0.2f})'
                             ''.format(roc_auc["macro"]))
    data.append(trace2)
    np.random.shuffle(colors)
    color_cycle = cycle(colors)

    for i, label, color in zip(class_indices, classes, color_cycle):
        trace3 = go.Scatter(x=fpr[i], y=tpr[i],
                            mode='lines',
                            line=dict(color=color, width=2),
                            name='ROC curve of {0} (area={1:0.2f})'
                                 ''.format(label, roc_auc[i]))
        data.append(trace3)

    trace4 = go.Scatter(x=[0, 1], y=[0, 1],
                        mode='lines',
                        line=dict(color='black', width=2, dash='dash'),
                        showlegend=False)
    data.append(trace4)

    fig = configure_figure(data, title, width, height)

    return fig


def configure_figure(data, title, width, height):
    layout = go.Layout(title=title,
                       xaxis=dict(title='False Positive Rate'),
                       yaxis=dict(title='True Positive Rate'),
                       width=width,
                       height=height,
                       margin=dict(
                           l=5,
                           r=5,
                           b=5,
                           t=5,
                           pad=5
                       ),
                       )
    fig = go.Figure(data=data, layout=layout)
    return fig
