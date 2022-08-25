from collections import defaultdict
from typing import Union

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import Index


def parse_labels(y_str: pd.Series, min_count: int = None, max_count: int = None,
                 labels_subset: Union[Index, ndarray] = None,
                 dropna: bool = False, delimiter: str = "|", verbose=False) -> pd.Series:
    if dropna:
        index = y_str.dropna().index
    else:
        index = y_str.index

    if delimiter and y_str.loc[index].map(lambda x: isinstance(x, str)).all():
        y_list = y_str.loc[index].str.split(delimiter)
    else:
        y_list = y_str.loc[index]

    if min_count or max_count or labels_subset is not None:
        selected_labels = select_labels(y_list, min_count=min_count, max_count=max_count)
        if labels_subset is not None:
            selected_labels = selected_labels.intersection(labels_subset)

        print(
            f"{y_str.name} num of labels selected: {len(selected_labels)} with min_count={min_count}") if verbose else None
    else:
        selected_labels = None

    y_df = y_list.map(lambda labels:
                      [item for item in labels if item in selected_labels] if selected_labels else labels \
                          if isinstance(labels, (list, np.ndarray)) else [])

    return y_df


def select_labels(y_list: pd.Series, min_count: Union[int, float] = None, max_count: int = None) -> pd.Index:
    """

    Args:
        y_list (pd.Series): A Series with values containing list of strings.
        min_count (float): If integer, then filter labels with at least `min_count` raw frequency. \
            If float, then filter labels annotated with at least `min_count` percentage of genes.

    Returns:
        labels_filter (pd.Index): filter
    """
    counts = defaultdict(lambda: 0)

    if isinstance(min_count, float) and min_count < 1.0:
        num_genes = y_list.shape[0]
        min_count = int(num_genes * min_count)
    elif min_count is None:
        min_count = 1

    # Filter a label if its label_counts is less than min_count
    for labels in y_list:
        if labels is None or not isinstance(labels, (str, float)): continue
        for label in labels:
            counts[label] = counts[label] + 1

    counts = pd.Series(counts)
    counts = counts[counts >= min_count]
    if max_count:
        counts = counts[counts <= max_count]

    return counts.index
