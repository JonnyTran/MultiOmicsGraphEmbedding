from typing import Union

import numpy as np
import pandas as pd
import tqdm
from numpy import ndarray
from pandas import Index


def filter_multilabel(y_str: pd.Series, min_count: int = None, max_count: int = None,
                      labels_subset: Union[Index, ndarray] = None,
                      dropna: bool = False, delimiter: str = "|", verbose=False) -> pd.Series:
    if dropna:
        index = y_str.dropna().index
    else:
        index = y_str.index

    if delimiter:
        y_list = y_str.loc[index].str.split(delimiter)
    else:
        y_list = y_str.loc[index]

    labels_filter = select_labels(y_list, min_count=min_count, max_count=max_count)
    if labels_subset is not None:
        labels_filter = labels_filter.intersection(labels_subset)

    print(f"{y_str.name} num of labels selected: {len(labels_filter)} with min_count={min_count}") if verbose else None

    y_df = y_list.map(lambda go_terms: \
                          [item for item in go_terms if item in labels_filter] \
                              if isinstance(go_terms, (list, np.ndarray)) else [])

    return y_df


def select_labels(y_list: pd.Series, min_count: Union[int, float], max_count: int = None) -> pd.Index:
    """

    Args:
        y_list (pd.Series): A Series with values containing list of strings.
        min_count (float): If integer, then filter labels with at least `min_count` raw frequency. \
            If float, then filter labels annotated with at least `min_count` percentage of genes.

    Returns:
        labels_filter (pd.Index): filter
    """
    label_counts = {}

    if isinstance(min_count, float) and min_count < 1.0:
        num_genes = y_list.shape[0]
        min_count = int(num_genes * min_count)
    elif min_count is None:
        min_count = 1

    # Filter a label if its label_counts is less than min_count
    for labels in tqdm.tqdm(y_list, desc=f"Count labels for {y_list.name} with >= {min_count} frequency."):
        if not isinstance(labels, list): continue
        for label in labels:
            label_counts[label] = label_counts.setdefault(label, 0) + 1

    label_counts = pd.Series(label_counts)
    label_counts = label_counts[label_counts >= min_count]
    if max_count:
        label_counts = label_counts[label_counts <= max_count]

    return label_counts.index
