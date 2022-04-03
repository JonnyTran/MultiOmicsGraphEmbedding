from typing import Union

import pandas as pd
import tqdm


def filter_multilabel(df: pd.DataFrame, column="go_id", min_count=2, max_count=None, label_subset: pd.Index = None,
                      dropna=False,
                      delimiter="|"):
    if dropna:
        nodes_index = df[[column]].dropna().index
    else:
        nodes_index = df.index

    if delimiter:
        labels = df.loc[nodes_index, column].str.split(delimiter)
    else:
        labels = df.loc[nodes_index, column]

    if min_count:
        labels_filter = filter_labels_by_count(labels, min_count=min_count, max_count=max_count)
        if label_subset is not None:
            labels_filter = labels_filter.intersection(label_subset)

        print(f"{df.index.name}'s {column} num of labels selected: {len(labels_filter)} with min_count={min_count}")
    else:
        labels_filter = labels

    y_labels = labels.map(
        lambda go_terms: [item for item in go_terms if item in labels_filter] \
            if type(go_terms) == list else [])

    return y_labels


def filter_labels_by_count(y_multilabel: pd.Series, min_count: Union[int, float], max_count: int = None):
    """

    Args:
        y_multilabel (pd.DataFrame): A dataframe with index for gene IDs and values for list of annotations.
        min_count (float): If integer, then filter labels with at least `min_count` raw frequency. \
            If float, then filter labels annotated with at least `min_count` percentage of genes.

    Returns:
        labels_filter (pd.Index): filter
    """
    label_counts = {}

    if isinstance(min_count, float) and min_count < 1.0:
        num_genes = y_multilabel.shape[0]
        min_count = int(num_genes * min_count)

    # Filter a label if its label_counts is less than min_count
    for labels in tqdm.tqdm(y_multilabel):
        if not isinstance(labels, list): continue
        for label in labels:
            label_counts[label] = label_counts.setdefault(label, 0) + 1

    label_counts = pd.Series(label_counts)
    label_counts = label_counts[label_counts >= min_count]
    if max_count:
        label_counts = label_counts[label_counts < max_count]

    return label_counts.index
