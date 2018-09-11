from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, roc_curve
import keras.backend as K

import numpy as np


def precision(y_true, y_pred):
    """Precision metric for keras models

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    y_pred = K.cast(y_pred < 0.5, y_true.dtype)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric for keras models

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    y_pred = K.cast(y_pred < 0.5, y_true.dtype)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(y_true, y_pred):
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def link_prediction_score(true_edges, pred_edges, directed=True, metrics=["precision", "recall"]):
    """
    Evaluate link prediction accuracy scores between true_edges, pred_edges. The

    :param true_edges: a list of ground truth edges. Each element is a tuple of two node gene names
    :param pred_edges: a list of predicted edges. Each element is a tuple of two node gene names
    :param directed: bool
    :param metrics: list of metrics ot compute
    :return:
    """
    if directed==False:
        true_edges = set([(edge[0], edge[1]) for edge in true_edges]).union(set([(edge[1], edge[0]) for edge in true_edges]))
        pred_edges = set([(edge[0], edge[1]) for edge in pred_edges]).union(
            set([(edge[1], edge[0]) for edge in pred_edges]))

    true_edges = set(true_edges)
    pred_edges = set(pred_edges)

    scores = {}
    if "precision" in metrics:
        scores["precision"] = float(len(pred_edges & true_edges)) / len(pred_edges)

    if "recall" in metrics:
        scores["recall"] = float(len(pred_edges & true_edges)) / len(true_edges)

    return scores


# Input: positive test/val edges, negative test/val edges, edge score matrix
# Output: ROC AUC score, ROC Curve (FPR, TPR, Thresholds), AP score
def get_roc_score(edges_pos, edges_neg, score_matrix):
    # Edge case
    if len(edges_pos) == 0 or len(edges_neg) == 0:
        return (None, None, None)

    # Store positive edge predictions, actual values
    preds_pos = []
    pos = []
    for edge in edges_pos:
        preds_pos.append(score_matrix[edge[0], edge[1]])
        pos.append(1)  # actual value (1 for positive)

    # Store negative edge predictions, actual values
    preds_neg = []
    neg = []
    for edge in edges_neg:
        preds_neg.append(score_matrix[edge[0], edge[1]])
        neg.append(0)  # actual value (0 for negative)

    # Calculate scores
    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    # roc_curve_tuple = roc_curve(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    # return roc_score, roc_curve_tuple, ap_score
    return roc_score, ap_score
