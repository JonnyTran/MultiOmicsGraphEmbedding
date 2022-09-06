from typing import Dict, Tuple, List, Any, Union

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer


def evaluate_classification(embedding, network, cv=5, node_label="Family", multilabel=False, classifier=None,
                            scoring=['precision_micro', 'recall_micro', "f1_micro"], verbose=False):
    nodelist = embedding.node_list
    genes_info = network.genes_info

    # Filter labels that have enough samples
    if multilabel:
        labels_to_test = [i for (i, v) in (genes_info[genes_info[node_label].notnull()][node_label].str.get_dummies(
            "|").sum(axis=0) >= cv).items() if v == True]
        nodes_with_label = [i for (i, v) in genes_info[genes_info[node_label].notnull()][node_label].str.contains(
            "|".join(labels_to_test)).items() if v == True]
    else:
        labels_to_test = [i for (i, v) in (genes_info[node_label].value_counts() >= cv).items() if v == True]
        nodes_with_label = genes_info[genes_info[node_label].isin(labels_to_test)].index

    nodelist = [node for node in nodelist if node in nodes_with_label]
    nodes_split_by_group = genes_info.loc[nodelist, node_label].str.split("|", expand=True)[0]
    print("labels_to_test", labels_to_test) if verbose > 1 else None
    print("# of labels with >cv samples:", len(labels_to_test), ", # of nodes to train/test:", len(nodelist)) if verbose else None

    X = embedding.get_embeddings(node_list=nodelist)
    assert len(nodelist) == X.shape[0]

    if multilabel:
        labels = genes_info.loc[nodelist, node_label].str.split("|", expand=False)
        y = MultiLabelBinarizer().fit_transform(labels.tolist())
    else:  # Multiclass classification (only single label each sample)
        y = genes_info.loc[nodelist, node_label].str.split("|", expand=True)[0]

    if classifier is not None:
        clf = classifier
    else:
        if multilabel:
            clf = KNeighborsClassifier(n_neighbors=3, weights="distance", algorithm="auto", metric="euclidean")
        else:
            clf = svm.LinearSVC(multi_class="ovr")

    scores = cross_validate(clf, X, y, groups=nodes_split_by_group, cv=cv, n_jobs=-2, scoring=scoring,
                            return_train_score=False)

    return scores


def compute_roc_auc_curve(targets, scores, class_indices, sample_weight=None):
    if isinstance(scores, pd.DataFrame):
        scores = scores.values
    if isinstance(targets, pd.DataFrame):
        scores = targets.values

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in class_indices:
        fpr[i], tpr[i], _ = roc_curve(targets[:, i], scores[:, i],
                                      sample_weight=sample_weight if sample_weight is not None else None)
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(targets.ravel(), scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return fpr, roc_auc, tpr


def compute_pr_curve(targets: np.ndarray, scores, class_indices: List[Any], sample_weight=None) \
        -> Tuple[Dict[str, Union[np.ndarray, float]], Dict[str, float], Dict[str, Union[np.ndarray, float]]]:
    if isinstance(scores, pd.DataFrame):
        scores = scores.values
    if isinstance(targets, pd.DataFrame):
        scores = targets.values

    precision = dict()
    recall = dict()
    avg_precision = dict()
    for i in class_indices:
        precision[i], recall[i], _ = precision_recall_curve(targets[:, i], scores[:, i],
                                                            sample_weight=sample_weight)

        avg_precision[i] = average_precision_score(targets[:, i], scores[:, i])

    # Compute micro-average ROC curve and ROC area
    precision["micro"], recall["micro"], _ = precision_recall_curve(targets.ravel(), scores.ravel())
    avg_precision["micro"] = average_precision_score(targets.ravel(), scores.ravel(), average="micro")
    avg_precision["macro"] = average_precision_score(targets, scores, average="macro")
    return precision, avg_precision, recall
