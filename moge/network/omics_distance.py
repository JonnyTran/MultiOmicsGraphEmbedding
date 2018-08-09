
import numpy as np
import pandas as pd
import networkx as nx

from sklearn.neighbors import DistanceMetric
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import pdist as scipy_pdist
from scipy.spatial.distance import squareform as squareform_

from Bio import pairwise2

from TCGAMultiOmics.multiomics import MultiOmicsData


def compute_expression_correlations(multi_omics_data: MultiOmicsData, modalities, pathologic_stages=[],
                                    histological_subtypes=[]):
    X_multiomics, y = multi_omics_data.load_data(modalities=modalities, pathologic_stages=pathologic_stages,
                                                 histological_subtypes=histological_subtypes)

    X_multiomics_concat = pd.concat([X_multiomics[m] for m in modalities], axis=1)
    X_multiomics_corr = np.corrcoef(X_multiomics_concat, rowvar=False)

    cols = X_multiomics_concat.columns
    X_multiomics_corr_df = pd.DataFrame(X_multiomics_corr, columns=cols, index=cols)

    return X_multiomics_corr_df

def compute_annotation_similarity(genes_info, modality, features=None, squareform=True, multiprocessing=True):
    if features is None:
        if modality == "GE":
            features = ["locus_type", "gene_family_id", "Transcript sequence"]
        elif modality == "MIR":
            features = ["miR family", "Mature sequence"]
        elif modality == "LNC":
            features = ["Transcript Type", "Transcript sequence"]

    gower_dists = gower_distance(genes_info.loc[:, features], multiprocessing)

    if squareform:
        return squareform_(np.subtract(1, gower_dists))
    else:
        return np.subtract(1, gower_dists)
    # return np.exp(-beta * gower_dists)

def gower_distance(X, multiprocessing=True, n_jobs=-2):
    """
    This function expects a pandas dataframe as input
    The data frame is to contain the features along the columns. Based on these features a
    distance matrix will be returned which will contain the pairwise gower distance between the rows
    All variables of object type will be treated as nominal variables and the others will be treated as
    numeric variables.
    Distance metrics used for:
    Nominal variables: Dice distance (https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)
    Numeric variables: Manhattan distance normalized by the range of the variable (https://en.wikipedia.org/wiki/Taxicab_geometry)
    """
    individual_variable_distances = []

    if multiprocessing:
        pdist = lambda X, metric: squareform_(pairwise_distances(X=X, metric=metric, n_jobs=n_jobs), checks=False)
    else:
        pdist = scipy_pdist # returns condensed dist matrix

    for column in X.columns:
        print("Gower's dissimilarity: Computing", column)
        feature = X.loc[:, column]
        if column == "gene_family_id" or column == "gene_family" or column == "locus_type":
            feature_dist = pdist(feature.str.get_dummies("|"), 'dice')

        elif column == "miR family":
            feature_dist = pdist(feature.str.get_dummies("/"), 'dice')

        elif column == "GO terms":
            feature_dist = pdist(feature.str.get_dummies(","), 'dice')

        elif column in ["Mature sequence", "Transcript sequence"]:
            feature_dist = pdist(feature.values.reshape((X.shape[0],-1)),
                                     metric=seq_global_alignment_pairwise_score)

            feature_dist = 1-feature_dist # Convert from similarity to dissimilarity

        elif feature.dtypes == np.object:
            feature_dist = pdist(pd.get_dummies(feature), 'dice')
        elif False: # if nominal numbers
            feature_dist = pdist(feature.values.reshape((X.shape[0],-1)), "manhattan") / np.ptp(feature.values)
        else:
            feature_dist = pdist(feature.values.reshape((X.shape[0],-1)), "euclidean")

        print(feature_dist)
        individual_variable_distances.append(feature_dist)

    pdists_mean = np.nanmean(np.array(individual_variable_distances), axis=0)

    return pdists_mean
    # return squareform(pdists_mean)

def seq_global_alignment_pairwise_score(u, v, truncate=False):
    if (type(u[0]) is str and type(v[0]) is str):
        return pairwise2.align.globalxx(u[0], v[0], score_only=True) / min(len(u[0]), len(v[0]))
    else:
        return np.nan


