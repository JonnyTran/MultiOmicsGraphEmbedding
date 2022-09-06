from itertools import cycle
from typing import Optional

import torch
from plotly import graph_objects as go
from torch import Tensor

from moge.criterion.classification import *
from .utils import colors


def plot_roc_curve_multiclass(y_test: pd.DataFrame, y_score, classes: (list, pd.Index), sample_weight=None,
                              title='ROC Curve (multi-class)', plot_classes=False,
                              width=800, height=700):
    if isinstance(y_test, pd.DataFrame) and classes is not None:
        class_indices = y_test.columns.get_indexer(classes)
    else:
        class_indices = range(y_test.shape[1])

    print("class_indices")

    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.values

    if isinstance(y_score, pd.DataFrame):
        y_score = y_score.values

    # Compute macro-average ROC curve and ROC area
    fpr, roc_auc, tpr = compute_roc_auc_curve(y_test, y_score, class_indices, sample_weight)

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in class_indices]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in class_indices:
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(class_indices)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    data = []
    trace1 = go.Scatter(x=fpr["micro"], y=tpr["micro"],
                        mode='lines',
                        line=dict(color='deeppink', width=2, dash='dot'),
                        name='micro-average ROC curve (area={0:0.2f})'.format(roc_auc["micro"]))
    data.append(trace1)

    trace2 = go.Scatter(x=fpr["macro"], y=tpr["macro"],
                        mode='lines',
                        line=dict(color='navy', width=2, dash='dot'),
                        name='macro-average ROC curve (area={0:0.2f})'.format(roc_auc["macro"]))
    data.append(trace2)
    np.random.shuffle(colors)
    color_cycle = cycle(colors)

    if plot_classes:
        for i, label, color in zip(class_indices, classes, color_cycle):
            trace3 = go.Scatter(x=fpr[i], y=tpr[i],
                                mode='lines',
                                line=dict(color=color, width=2),
                                name='ROC curve of {0} (area={1:0.2f})'.format(label, roc_auc[i]))
            data.append(trace3)

    trace4 = go.Scatter(x=[0, 1], y=[0, 1],
                        mode='lines',
                        line=dict(color='black', width=2, dash='dash'),
                        showlegend=False)
    data.append(trace4)

    fig = configure_figure(data, title, width, height, x_label='False Positive Rate', y_label='True Positive Rate')

    return fig


def plot_roc_curve(y_test, y_score, n_classes, sample_weight=None, width=700, height=700,
                   title='Receiver operating characteristic example'):
    # Compute ROC curve and ROC area for each class
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.values

    if isinstance(y_score, pd.DataFrame):
        y_score = y_score.values

    fpr, roc_auc, tpr = compute_roc_auc_curve(y_test, y_score, range(n_classes), sample_weight)

    trace1 = go.Scatter(x=fpr[2], y=tpr[2],
                        mode='lines',
                        line=dict(color='darkorange', width=2),
                        name='ROC curve (area = %0.2f)' % roc_auc[2])

    trace2 = go.Scatter(x=[0, 1], y=[0, 1],
                        mode='lines',
                        line=dict(color='navy', width=2, dash='dash'),
                        showlegend=False)

    fig = configure_figure([trace1, trace2], title, width, height, x_label='False Positive Rate',
                           y_label='True Positive Rate')
    return fig


def plot_pr_curve_multiclass(targets: pd.DataFrame, scores: Union[np.ndarray, Tensor],
                             classes: Optional[Union[list, pd.Index]] = None,
                             sample_weight=None,
                             plot_classes=False,
                             title='PR Curve (multi-class)', width=800, height=700):
    if isinstance(targets, pd.DataFrame) and classes is not None:
        class_indices = targets.columns.get_indexer(classes)
    else:
        class_indices = range(targets.shape[1])
    if classes is None:
        classes = class_indices

    if isinstance(targets, pd.DataFrame):
        targets = targets.astype(int).values
    elif isinstance(targets, Tensor):
        targets = targets.type(torch.int).numpy()

    if isinstance(scores, pd.DataFrame):
        scores = scores.values
    elif isinstance(scores, Tensor):
        scores = scores.numpy()

    # Compute macro-average PR curve and PR area
    precision, avg_precision, recall = compute_pr_curve(targets, scores, class_indices=class_indices,
                                                        sample_weight=sample_weight)

    # First aggregate all false positive rates
    all_precision = np.unique(np.concatenate([precision[i] for i in class_indices]))

    # Then interpolate all PR curves at this points
    mean_tpr = np.zeros_like(all_precision)
    for i in class_indices:
        mean_tpr += np.interp(all_precision, precision[i], recall[i])

    # Finally average it and compute AUC
    mean_tpr /= len(classes)

    precision["macro"] = all_precision
    recall["macro"] = mean_tpr

    # Plot all ROC curves
    data = []
    trace1 = go.Scatter(x=recall["micro"], y=precision["micro"],
                        mode='lines', line=dict(color='deeppink', width=2, dash='dot'),
                        name='micro-average PR curve (area={0:0.2f})'.format(avg_precision["micro"]))
    data.append(trace1)

    trace2 = go.Scatter(x=recall["macro"], y=precision["macro"],
                        mode='lines', line=dict(color='navy', width=2, dash='dot'),
                        name='macro-average PR curve (area={0:0.2f})'.format(avg_precision["macro"]))
    data.append(trace2)
    np.random.shuffle(colors)
    color_cycle = cycle(colors)

    if plot_classes:
        for i, label, color in zip(class_indices, classes, color_cycle):
            trace3 = go.Scatter(x=precision[i], y=recall[i],
                                mode='lines', line=dict(color=color, width=2),
                                name='PR curve of {0} (area={1:0.2f})'.format(label, avg_precision[i]))
            data.append(trace3)

    trace4 = go.Scatter(x=[0, 1], y=[0, 1],
                        mode='lines', line=dict(color='black', width=2, dash='dash'),
                        showlegend=False)
    data.append(trace4)

    fig = configure_figure(data, x_label="Recall", y_label="Precision", width=width, height=height, title=title)

    return fig


def configure_figure(data, x_label='False Positive Rate', y_label='True Positive Rate', **kwargs):
    layout = go.Layout(xaxis=dict(title=x_label),
                       yaxis=dict(title=y_label),
                       legend=dict(x=0.4, y=0.0),
                       **kwargs,
                       )
    fig = go.Figure(data=data, layout=layout)

    return fig
