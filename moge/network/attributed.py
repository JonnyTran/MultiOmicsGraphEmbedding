import traceback
from typing import List, Dict, Union

import numpy as np
import pandas as pd
from moge.network.base import Network
from moge.network.base import SEQUENCE_COL
from moge.network.utils import select_labels
from sklearn import preprocessing

import openomics
from openomics.transforms.agg import concat_uniques

EPSILON = 1e-16
MODALITY_COL = "omic"


class AttributedNetwork(Network):
    def __init__(self, multiomics: openomics.MultiOmics, annotations=True, **kwargs) -> None:
        """
        Handles the MultiOmics attributes associated to the network(s).

        :param multiomics: an openomics.MultiOmics instance.
        :param annotations: default True. Whether to run annotations processing.
        :param kwargs: args to pass to Network() constructor.
        """
        self.multiomics = multiomics

        # Process network & node_list
        super().__init__(**kwargs)

        # Process node attributes
        if annotations:
            self.process_annotations()
            self.process_feature_tranformer()

    def process_annotations(self):
        annotations_list = []

        for modality in self.modalities:
            annotation = self.multiomics[modality].get_annotations()
            annotation[MODALITY_COL] = modality
            annotations_list.append(annotation)

        self.annotations = pd.concat(annotations_list, join="inner", copy=True)
        assert type(
            self.annotations.index) != pd.MultiIndex, "Annotation index must be a pandas.Index type and not a MultiIndex."

        # self.annotations = self.annotations[~self.annotations.index.duplicated(keep='first')]
        self.annotations = self.annotations.groupby(self.annotations.index).agg(
            {k: concat_uniques for k in self.annotations.columns})

        print("Annotation columns:", self.annotations.columns.tolist())

    def process_feature_tranformer(self, columns=None, delimiter="\||;", labels_subset=None, min_count=0,
                                   verbose=False):
        """
        For each of the annotation column, create a sklearn label binarizer. If the column data is delimited, a MultiLabelBinarizer
        is used to convert a list of labels into a vector.
        :param delimiter (str): default "|".
        :param min_count (int): default 0. Remove labels with frequency less than this. Used for classification or train/test stratification tasks.

        Args:
            columns ():
        """
        self.delimiter = delimiter

        if not hasattr(self, "feature_transformer"):
            self.feature_transformer = {}

        df = self.annotations
        if columns:
            df.filter(columns, axis='columns')
        transformers = self.get_feature_transformers(df, node_list=self.node_list, labels_subset=labels_subset,
                                                     min_count=min_count,
                                                     delimiter=delimiter, verbose=verbose)
        self.feature_transformer.update(transformers)

    @classmethod
    def get_feature_transformers(cls, annotation: pd.DataFrame,
                                 labels_subset: List[str] = None,
                                 min_count: int = 0,
                                 delimiter="\||;",
                                 verbose=False) \
            -> Dict[str, Union[preprocessing.MultiLabelBinarizer, preprocessing.StandardScaler]]:
        """
        :param annotation: a pandas DataFrame
        :param node_list: list of nodes. Indexes the annotation DataFrame
        :param labels_subset: str or list of str for the labels to filter by min_count
        :param min_count: minimum frequency of label to keep
        :param delimiter: default "\||;", delimiter ('|' or ';') to split strings
        :return: dict of feature transformers
        """
        transformers: Dict[str, preprocessing.MultiLabelBinarizer] = {}
        for col in annotation.columns:
            if col == SEQUENCE_COL:
                continue

            values: pd.Series = annotation[col].dropna(axis=0)
            if values.map(type).nunique() > 1:
                print(f"WARN: {col} has more than 1 dtypes: {values.map(type).unique()}")

            try:
                if annotation[col].dtypes == np.object and (annotation[col].dropna().map(type) == str).all():
                    transformers[col] = preprocessing.MultiLabelBinarizer()

                    if annotation[col].str.contains(delimiter, regex=True).any():
                        print("INFO: Label {} (of str split by '{}') transformed by MultiLabelBinarizer".format(col,
                                                                                                                delimiter)) if verbose else None
                        values = values.str.split(delimiter)
                        values = values.map(
                            lambda x: [term.strip() for term in x if len(term) > 0] if isinstance(x, list) else x)

                    if labels_subset is not None and col in labels_subset and min_count:
                        labels_subset = select_labels(values, min_count=min_count)
                        values = values.map(lambda labels: [item for item in labels if item not in labels_subset])

                    transformers[col].fit(values)

                elif annotation[col].dtypes == int or annotation[col].dtypes == float:
                    print(
                        "INFO: Label {} (of int/float) is transformed by StandardScaler".format(
                            col)) if verbose else None
                    transformers[col] = preprocessing.StandardScaler()

                    values = values.dropna().to_numpy()
                    transformers[col].fit(values.reshape(-1, 1))

                else:
                    print("INFO: Label {} is transformed by MultiLabelBinarizer".format(col)) if verbose else None
                    transformers[col] = preprocessing.MultiLabelBinarizer()
                    values = values[~values.map(type).isin({str, float, int, bool, type(None)})]
                    transformers[col].fit(values)

                if hasattr(transformers[col], 'classes_') and "" in transformers[col].classes_:
                    print(f"removed '' from classes in {col}")
                    transformers[col].classes_ = np.delete(transformers[col].classes_,
                                                           np.where(transformers[col].classes_ == "")[0])
            except Exception as e:
                print(e)
                traceback.print_exc()
                continue

        return transformers

    def get_labels_color(self, label, go_id_colors, child_terms=True, fillna="#e5ecf6", label_filter=None):
        """
        Filter the gene GO annotations and assign a color for each term given :param go_id_colors:.
        """
        if hasattr(self, "all_annotations"):
            labels = self.all_annotations[label].copy(deep=True)
        else:
            labels = self.annotations[label].copy(deep=True)

        if labels.str.contains("\||;", regex=True).any():
            labels = labels.str.split("\||;")

        if label_filter is not None:
            # Filter only annotations in label_filter
            if not isinstance(label_filter, set): label_filter = set(label_filter)
            labels = labels.map(lambda x: [term for term in x if term in label_filter] if x and len(x) > 0 else None)

        # Filter only annotations with an associated color
        labels = labels.map(lambda x: [term for term in x if term in go_id_colors.index] if x and len(x) > 0 else None)

        # For each node select one term
        labels = labels.map(lambda x: sorted(x)[-1 if child_terms else 0] if x and len(x) >= 1 else None)
        label_color = labels.map(go_id_colors)
        if fillna:
            label_color.fillna("#e5ecf6", inplace=True)
        return label_color


