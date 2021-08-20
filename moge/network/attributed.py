import numpy as np
import openomics
import pandas as pd
from openomics.utils.df import concat_uniques
from sklearn import preprocessing
from typing import List, Union

from moge.data.sequences import SEQUENCE_COL
from moge.network.semantic_similarity import compute_expression_correlation, compute_annotation_affinities
from moge.network.base import Network

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
        super(AttributedNetwork, self).__init__(**kwargs)

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

    def process_feature_tranformer(self, delimiter="\||;", filter_label=None, min_count=0, verbose=False):
        """
        For each of the annotation column, create a sklearn label binarizer. If the column data is delimited, a MultiLabelBinarizer
        is used to convert a list of labels into a vector.
        :param delimiter (str): default "|".
        :param min_count (int): default 0. Remove labels with frequency less than this. Used for classification or train/test stratification tasks.
        """
        self.delimiter = delimiter
        self.feature_transformer = self.get_feature_transformers(self.annotations, self.node_list, filter_label,
                                                                 min_count, delimiter, verbose=verbose)

    @classmethod
    def get_feature_transformers(cls, annotation: pd.DataFrame, node_list, filter_label: List[str] = None,
                                 min_count: int = 0,
                                 delimiter="\||;",
                                 verbose=False):
        """
        :param annotation: a pandas DataFrame
        :param node_list: list of nodes. Indexes the annotation DataFrame
        :param filter_label: str or list of str for the labels to filter by min_count
        :param min_count: minimum frequency of label to keep
        :param delimiter: default "\||;", delimiter ('|' or ';') to split strings
        :return: dict of feature transformers
        """
        feature_transformers = {}
        for col in annotation.columns:
            if col == SEQUENCE_COL:
                continue

            if annotation[col].dtypes == np.object:
                feature_transformers[col] = preprocessing.MultiLabelBinarizer()

                if annotation[col].str.contains(delimiter, regex=True).any():
                    print("INFO: Label {} (of str split by '{}') transformed by MultiLabelBinarizer".format(col,
                                                                                                            delimiter)) if verbose else None
                    features = annotation.loc[node_list, col].dropna(axis=0).str.split(delimiter)
                    features = features.map(
                        lambda x: [term.strip() for term in x if len(term) > 0] if isinstance(x, list) else x)
                else:
                    print("INFO: Label {} (of str) is transformed by MultiLabelBinarizer".format(
                        col)) if verbose else None
                    features = annotation.loc[node_list, col].dropna(axis=0)

                if filter_label is not None and col in filter_label and min_count:
                    labels_filter = filter_labels_by_count(features, min_count=min_count)
                    features = features.map(lambda labels: [item for item in labels if item not in labels_filter])
                feature_transformers[col].fit(features)

            elif annotation[col].dtypes == int or annotation[col].dtypes == float:
                print(
                    "INFO: Label {} (of int/float) is transformed by StandardScaler".format(col)) if verbose else None
                feature_transformers[col] = preprocessing.StandardScaler()
                features = annotation.loc[node_list, col].dropna(axis=0)
                feature_transformers[col].fit(features.to_numpy().reshape(-1, 1))

            else:
                print("INFO: Label {} is transformed by MultiLabelBinarizer".format(col)) if verbose else None
                feature_transformers[col] = preprocessing.MultiLabelBinarizer()
                features = annotation.loc[node_list, col].dropna(axis=0)
                feature_transformers[col].fit(features.to_numpy().reshape(-1, 1))

        return feature_transformers

    def get_correlation_edges(self, modality, node_list, threshold=0.7):
        # Filter similarity adj by correlation
        correlations = compute_expression_correlation(self.multiomics, modalities=[modality],
                                                      node_list=node_list, absolute_corr=True,
                                                      return_distance=False,
                                                      squareform=True)

        # Selects positive edges with high affinity in the affinity matrix
        similarity_filtered = np.triu(correlations >= threshold, k=1)  # A True/False matrix
        edgelist = [(node_list[x], node_list[y], {"weight": correlations.iloc[x, y]}) for x, y in
                    zip(*np.nonzero(similarity_filtered))]

        return edgelist

    def add_affinity_edges(self, modality, node_list, features=None, weights=None, nanmean=True,
                           similarity_threshold=0.7, dissimilarity_threshold=0.1,
                           negative_sampling_ratio=2.0, max_positive_edges=None,
                           tissue_expression=False, histological_subtypes=[],
                           pathologic_stages=[],
                           epsilon=EPSILON, tag="affinity"):
        """
        Computes similarity measures between genes within the same modality, and add them as undirected edges to the
network if the similarity measures passes the threshold

        :param modality: E.g. ["GE", "MIR", "LNC"]
        :param similarity_threshold: a hard-threshold to select positive edges with affinity value more than it
        :param dissimilarity_threshold: a hard-threshold to select negative edges with affinity value less than
        :param negative_sampling_ratio: the number of negative edges in proportion to positive edges to select
        :param histological_subtypes: the patients' cancer subtype group to calculate correlation from
        :param pathologic_stages: the patient's cancer stage group to calculate correlations from
        """
        annotations = self.multiomics[modality].get_annotations()

        annotation_affinities_df = pd.DataFrame(
            data=compute_annotation_affinities(annotations, node_list=node_list, modality=modality,
                                               correlation_dist=None, nanmean=nanmean,
                                               features=features, weights=weights, squareform=True),
            index=node_list)

        # Selects positive edges with high affinity in the affinity matrix
        similarity_filtered = np.triu(annotation_affinities_df >= similarity_threshold, k=1)  # A True/False matrix
        sim_edgelist_ebunch = [(node_list[x], node_list[y], annotation_affinities_df.iloc[x, y]) for x, y in
                               zip(*np.nonzero(similarity_filtered))]
        # Sample
        if max_positive_edges is not None:
            sample_indices = np.random.choice(a=range(len(sim_edgelist_ebunch)),
                                              size=min(max_positive_edges, len(sim_edgelist_ebunch)), replace=False)
            sim_edgelist_ebunch = [(u, v, d) for i, (u, v, d) in enumerate(sim_edgelist_ebunch) if i in sample_indices]
            self.G_u.add_weighted_edges_from(sim_edgelist_ebunch, type="u", tag=tag)
        else:
            self.G_u.add_weighted_edges_from(sim_edgelist_ebunch, type="u", tag=tag)

        print(len(sim_edgelist_ebunch), "undirected positive edges (type='u') added.")

        # Select negative edges at affinity close to zero in the affinity matrix
        max_negative_edges = int(negative_sampling_ratio * len(sim_edgelist_ebunch))
        dissimilarity_filtered = np.triu(annotation_affinities_df <= dissimilarity_threshold, k=1)

        dissimilarity_index_rows, dissimilarity_index_cols = np.nonzero(dissimilarity_filtered)
        sample_indices = np.random.choice(a=dissimilarity_index_rows.shape[0],
                                          size=min(max_negative_edges, dissimilarity_index_rows.shape[0]),
                                          replace=False)
        # adds 1e-8 to keeps from 0.0 edge weights, which doesn't get picked up in nx.adjacency_matrix()
        dissim_edgelist_ebunch = [(node_list[x], node_list[y], min(annotation_affinities_df.iloc[x, y], epsilon)) for
                                  i, (x, y) in
                                  enumerate(zip(dissimilarity_index_rows[sample_indices],
                                                dissimilarity_index_cols[sample_indices])) if i < max_negative_edges]
        self.G_u.add_weighted_edges_from(dissim_edgelist_ebunch, type="u_n", tag=tag)

        print(len(dissim_edgelist_ebunch), "undirected negative edges (type='u_n') added.")
        return annotation_affinities_df

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


def filter_multilabel(df: pd.DataFrame, column="go_id", min_count=2, label_subset: pd.Index = None, dropna=False,
                      delimiter="|"):
    if dropna:
        nodes_index = df[[column]].dropna().index
    else:
        nodes_index = df.index

    if df.loc[nodes_index, column].dtypes == np.object and \
            df.loc[nodes_index, column].str.contains(delimiter, regex=True).any():
        annotations_list = df.loc[nodes_index, column].str.split(delimiter)
    else:
        annotations_list = df.loc[nodes_index, column]

    if min_count:
        labels_filter = filter_labels_by_count(annotations_list, min_count=min_count)
        if label_subset is not None:
            labels_filter = labels_filter.intersection(label_subset)

        print(f"Column {column} filtered: {len(labels_filter)} with min_count={min_count}")
    else:
        labels_filter = annotations_list

    y_labels = annotations_list.map(
        lambda go_terms: [item for item in go_terms \
                          if item not in labels_filter] \
            if type(go_terms) == list else [])

    return y_labels


def filter_labels_by_count(annotation: pd.Series, min_count: Union[int, float]):
    """

    Args:
        annotation (pd.DataFrame): A dataframe with index for gene IDs and values for list of annotations.
        min_count (float): If integer, then filter labels with at least `min_count` raw frequency. If float, then filter labels annotated with at least `min_count` percentage of genes.

    Returns:
        labels_filter (pd.Index): filter
    """
    label_counts = {}

    if isinstance(min_count, float) and min_count < 1.0:
        num_genes = annotation.shape[0]
        min_count = int(num_genes * min_count)

    # Filter a label if its label_counts is less than min_count
    for items in annotation:
        if not isinstance(items, list): continue
        for item in items:
            label_counts[item] = label_counts.setdefault(item, 0) + 1

    label_counts = pd.Series(label_counts)
    labels_filter = label_counts[label_counts < min_count].index
    return labels_filter
