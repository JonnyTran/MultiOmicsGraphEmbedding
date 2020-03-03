import numpy as np
import pandas as pd
from sklearn import preprocessing

from moge.network.base import Network
from moge.network.semantic_similarity import compute_expression_correlation_dists, compute_annotation_affinities
from moge.network.train_test_split import get_labels_filter

EPSILON = 1e-16


class AttributedNetwork(Network):
    def __init__(self, multiomics, process_annotations=True, **kwargs) -> None:
        self.multiomics = multiomics
        if process_annotations:
            self.process_annotations()
            self.process_feature_tranformer()

        super(AttributedNetwork, self).__init__(**kwargs)

    def process_annotations(self):
        annotations_list = []

        for modality in self.modalities:
            annotation = self.multiomics[modality].get_annotations()
            annotation["omic"] = modality
            annotations_list.append(annotation)

        self.annotations = pd.concat(annotations_list, join="inner", copy=True)
        assert type(
            self.annotations.index) != pd.MultiIndex, "Annotation index must be a pandas.Index type and not a MultiIndex."
        self.annotations = self.annotations[~self.annotations.index.duplicated(keep='first')]
        print("Annotation columns:", self.annotations.columns.tolist())

    def process_feature_tranformer(self, min_count=0):
        self.feature_transformer = {}
        for label in self.annotations.columns:
            if label == 'Transcript sequence':
                continue

            if self.annotations[label].dtypes == np.object and self.annotations[label].str.contains("|").any():
                self.feature_transformer[label] = preprocessing.MultiLabelBinarizer()
                features = self.annotations.loc[self.node_list, label].dropna(axis=0).str.split("|")
                if min_count:
                    labels_filter = get_labels_filter(self, features.index, label, min_count=min_count)
                    features = features.map(lambda labels: [item for item in labels if item not in labels_filter])
                self.feature_transformer[label].fit(features)

            elif self.annotations[label].dtypes == int or self.annotations[label].dtypes == float:
                self.feature_transformer[label] = preprocessing.StandardScaler()
                features = self.annotations.loc[self.node_list, label].dropna(axis=0)
                self.feature_transformer[label].fit(features.to_numpy().reshape(-1, 1))

            else:
                self.feature_transformer[label] = preprocessing.MultiLabelBinarizer()
                features = self.annotations.loc[self.node_list, label].dropna(axis=0)
                self.feature_transformer[label].fit(features.to_numpy().reshape(-1, 1))

    def add_undirected_edges_from_attibutes(self, modality, node_list, features=None, weights=None,
                                            nanmean=True,
                                            similarity_threshold=0.7, dissimilarity_threshold=0.1,
                                            negative_sampling_ratio=2.0, max_positive_edges=None,
                                            compute_correlation=True, tissue_expression=False, histological_subtypes=[],
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

        # Filter similarity adj by correlation
        if compute_correlation:
            correlation_dist = compute_expression_correlation_dists(self.multiomics, modalities=[modality],
                                                                    node_list=node_list, absolute_corr=True,
                                                                    return_distance=True,
                                                                    histological_subtypes=histological_subtypes,
                                                                    pathologic_stages=pathologic_stages,
                                                                    squareform=False,
                                                                    tissue_expression=tissue_expression)
        else:
            correlation_dist = None

        annotation_affinities_df = pd.DataFrame(
            data=compute_annotation_affinities(annotations, node_list=node_list, modality=modality,
                                               correlation_dist=correlation_dist, nanmean=nanmean,
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


class MultiplexAttributedNetwork(AttributedNetwork):

    def __init__(self, multiomics, process_annotations=True) -> None:
        super().__init__(multiomics, process_annotations)

    def process_annotations(self):
        self.annotations = {}
        for modality in self.modalities:
            annotation = self.multiomics[modality].get_annotations()

            self.annotations[modality] = annotation

        print("Annotation columns:",
              {modality: annotations.columns.tolist() for modality, annotations in self.annotations.items()})
