from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, roc_curve

import numpy as np


def link_prediction_score(true_edges, pred_edges, directed=True, metrics=["precision", "recall"]):
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
