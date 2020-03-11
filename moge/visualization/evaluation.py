from itertools import cycle

import numpy as np
from plotly import graph_objects as go
from sklearn.metrics import roc_curve, auc


def plot_roc_curve(y_test, y_score, n_classes, sample_weight=None):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i],
                                      sample_weight=sample_weight if sample_weight != None else None)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel(),
                                              sample_weight=sample_weight if sample_weight != None else None)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    trace1 = go.Scatter(x=fpr[2], y=tpr[2],
                        mode='lines',
                        line=dict(color='darkorange', width=2),
                        name='ROC curve (area = %0.2f)' % roc_auc[2]
                        )

    trace2 = go.Scatter(x=[0, 1], y=[0, 1],
                        mode='lines',
                        line=dict(color='navy', width=2, dash='dash'),
                        showlegend=False)

    layout = go.Layout(title='Receiver operating characteristic example',
                       xaxis=dict(title='False Positive Rate'),
                       yaxis=dict(title='True Positive Rate'),
                       width=700,
                       height=700
                       )

    fig = go.Figure(data=[trace1, trace2], layout=layout)
    return fig


def plot_roc_curve_multiclass(y_test, y_score, n_classes, sample_weight=None, title='ROC Curve (multi-class)'):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i],
                                      sample_weight=sample_weight if sample_weight != None else None)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel(),
                                              sample_weight=sample_weight if sample_weight != None else None)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    data = []
    trace1 = go.Scatter(x=fpr["micro"], y=tpr["micro"],
                        mode='lines',
                        line=dict(color='deeppink', width=2, dash='dot'),
                        name='micro-average ROC curve (area = {0:0.2f})'
                             ''.format(roc_auc["micro"]))
    data.append(trace1)

    trace2 = go.Scatter(x=fpr["macro"], y=tpr["macro"],
                        mode='lines',
                        line=dict(color='navy', width=2, dash='dot'),
                        name='macro-average ROC curve (area = {0:0.2f})'
                             ''.format(roc_auc["macro"]))
    data.append(trace2)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        trace3 = go.Scatter(x=fpr[i], y=tpr[i],
                            mode='lines',
                            line=dict(color=color, width=2),
                            name='ROC curve of class {0} (area = {1:0.2f})'
                                 ''.format(i, roc_auc[i]))
        data.append(trace3)

    trace4 = go.Scatter(x=[0, 1], y=[0, 1],
                        mode='lines',
                        line=dict(color='black', width=2, dash='dash'),
                        showlegend=False)

    layout = go.Layout(title=title,
                       xaxis=dict(title='False Positive Rate'),
                       yaxis=dict(title='True Positive Rate'),
                       width=700,
                       height=700
                       )

    fig = go.Figure(data=data, layout=layout)

    return fig
