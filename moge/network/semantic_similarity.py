import numpy as np
import pandas as pd
from Bio import pairwise2
from openomics import MultiOmics
from scipy.spatial.distance import pdist as scipy_pdist
from scipy.spatial.distance import squareform as squareform_
from sklearn.metrics.pairwise import pairwise_distances


def compute_annotation_affinities(genes_info, node_list, modality=None, correlation_dist=None, features=None,
                                  weights=None,
                                  squareform=True, nanmean=True,
                                  multiprocessing=True):
    if features is None and modality is not None:
        if modality == "MessengerRNA":
            features = ["locus_type", "gene_family_id", "GO Terms", "location", "Disease association",
                        "Transcript sequence"]
        elif modality == "MicroRNA":
            features = ["Family", "location", "GO Terms", "Rfams", "Disease association", "Transcript sequence"]
        elif modality == "LncRNA":
            features = ["Transcript type", "Strand", "tag", "GO Terms", "Rfams", "Disease association",
                        "Transcript sequence"]

    if nanmean:
        agg_func = lambda x: np.nanmean(x, axis=0)
    else:
        agg_func = lambda x: np.average(x, axis=0, weights=weights)

    gower_dists = gower_distance(genes_info.loc[node_list, features], agg_func=agg_func,
                                 correlation_dist=correlation_dist,
                                 multiprocessing=multiprocessing)  # Returns a condensed distance matrix

    if squareform:
        return squareform_(np.subtract(1, gower_dists))
    else:
        return np.subtract(1, gower_dists)  # Turns distance to similarity measure
    # return np.exp(-beta * gower_dists)


def compute_expression_correlation(multiomics: MultiOmics, modalities, node_list, absolute_corr=True,
                                   return_distance=True,
                                   pathologic_stages=[], histological_subtypes=[], squareform=True,
                                   tissue_expression=False):
    # Only works with TCGA expression data
    X_multiomics, y = multiomics.load_data(omics=modalities, pathologic_stages=pathologic_stages,
                                           histological_subtypes=histological_subtypes)

    # Remove duplicate index and columns
    for modality in modalities:
        X_multiomics[modality] = X_multiomics[modality].loc[
            ~X_multiomics[modality].index.duplicated(keep='first'), ~X_multiomics[modality].columns.duplicated(
                keep='first')]

    # TODO temporary implementation of using tissue expressions instead of TCGA expression
    if tissue_expression is not False:
        X_multiomics[modalities[0]] = pd.DataFrame(columns=tissue_expression.columns,
                                                   index=X_multiomics[modalities[0]].columns)
        X_multiomics[modalities[0]].fillna(tissue_expression, inplace=True)
        X_multiomics[modalities[0]] = X_multiomics[modalities[0]].T

    # Concatenate all modalities
    X_multiomics_concat = pd.concat([X_multiomics[m] for m in modalities], axis=1)

    # Calculate the correlation DISTANCE between all genes/transcripts, keeping shape with null values
    X_multiomics_corr_dists = pairwise_distances(X_multiomics_concat.T,
                                                 metric="correlation",
                                                 force_all_finite="allow_nan")

    # Filter to only correlations between nodes in node_lists
    cols = X_multiomics_concat.columns
    X_multiomics_corr_df = pd.DataFrame(X_multiomics_corr_dists, columns=cols, index=cols)
    X_multiomics_corr_df = X_multiomics_corr_df.loc[
        ~X_multiomics_corr_df.index.duplicated(keep='first'), ~X_multiomics_corr_df.columns.duplicated(keep='first')]
    X_multiomics_corr_df = X_multiomics_corr_df.filter(items=node_list)
    X_multiomics_corr_df = X_multiomics_corr_df.filter(items=node_list, axis=0)

    if absolute_corr:
        X_multiomics_corr_df = 1 - X_multiomics_corr_df
        X_multiomics_corr_df = np.abs(X_multiomics_corr_df)
        X_multiomics_corr_df = 1 - X_multiomics_corr_df

    if return_distance == False:
        X_multiomics_corr_df = 1 - X_multiomics_corr_df

    if squareform:
        return X_multiomics_corr_df
    else:
        return squareform_(X_multiomics_corr_df, checks=False) # Returns condensed distance matrix


def gower_distance(X: pd.DataFrame, agg_func=None, correlation_dist=None, multiprocessing=True, n_jobs=-2,
                   verbose=False):
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
    individual_variable_dists = []
    if multiprocessing:
        pdist = lambda X, metric: squareform_(pairwise_distances(X=X, metric=metric, n_jobs=n_jobs,
                                                                 force_all_finite='allow-nan'),
                                              checks=False)
    else:
        pdist = scipy_pdist # returns condensed dist matrix

    for column in X.columns:
        feature = X.loc[:, column]
        print("Gower's dissimilarity: Computing", column, ", dtype:", feature.dtypes, ", shape:",
              feature.shape) if verbose else None

        if column in ["gene_family_id", "gene_family", "locus_type", "Transcript type", "tag"]:
            print("Dice distance") if verbose else None
            feature_dist = pdist(feature.str.get_dummies("|"), 'dice')

        elif column == "miR family" or column == "Family":
            print("Dice distance") if verbose else None
            feature_dist = pdist(feature.str.get_dummies("/"), 'dice')

        elif column == "GO terms" or column == "Rfams":
            print("Dice distance") if verbose else None
            feature_dist = pdist(feature.str.get_dummies("|"), 'dice')

        elif column == "Disease association":
            print("Dice distance") if verbose else None
            feature_dist = pdist(feature.str.get_dummies("|"), 'dice')

        elif "sequence" in column:
            print(f"Global alignment seq score (maxlen={100})") if verbose else None
            # Note: If doesn't work, modify _pairwise_callable Line 1083  # X, Y = check_pairwise_arrays(X, Y)
            feature_dist = pdist(feature.values.reshape((X.shape[0], -1)), seq_global_alignment_pairwise_score)
            feature_dist = 1 - feature_dist  # Convert from similarity to dissimilarity

        elif column == "Location": # LNC Locations
            print("Location split to Chromosome, start, end") if verbose else None
            location_features = feature.str.split("[:-]", expand=True).filter(items=[0, 1])
            hierarchical_columns = ["Chromosome", "start"]
            location_features.columns = hierarchical_columns
            location_features["start"] = location_features["start"].astype(np.float64)
            location_features["end"] = location_features["end"].astype(np.float64) # TODO Add bp region length

            feature_dist = gower_distance(location_features, agg_func=hierarchical_distance_aggregate_score,
                                          multiprocessing=True)

        elif column == "location": # GE Locations
            print("Location split to Chromosome, arm, region") if verbose else None
            location_features = feature.str.split("[pq.]", expand=True).filter(items=[0, 1])
            location_features.columns = ["Chromosome", "region"]
            location_features["arm"] = feature.str.extract(r'(?P<arm>[pq])', expand=True)
            location_features["band"] = feature.str.split("[pq.-]", expand=True)[2]
            location_features = location_features[["Chromosome", "arm", "region", "band"]] # TODO Add band #
            # print(location_features)
            feature_dist = gower_distance(location_features, agg_func=hierarchical_distance_aggregate_score,
                                          multiprocessing=True)

        elif feature.dtypes == np.object: # TODO Use Categorical dtypes later
            print("Dice distance") if verbose else None
            feature_dist = pdist(pd.get_dummies(feature), 'dice')

        elif feature.dtypes == int:
            print("Manhattan distance (normalized ptp)") if verbose else None
            feature_dist = scipy_pdist(feature.values.reshape((X.shape[0],-1)), "manhattan") / \
                           (np.nanmax(feature.values) - np.nanmin(feature.values))
        elif feature.dtypes == float:
            print("Euclidean distance (normalized ptp)") if verbose else None
            feature_dist = scipy_pdist(feature.values.reshape((X.shape[0],-1)), "euclidean") / \
                           (np.nanmax(feature.values) - np.nanmin(feature.values))
        else:
            raise Exception("Invalid column dtype")

        individual_variable_dists.append(feature_dist)

    if correlation_dist is not None:
        print("Correlation distance", correlation_dist.shape) if verbose else None
        individual_variable_dists.append(correlation_dist)

    if agg_func is None:
        agg_func = lambda x: np.nanmean(x, axis=0)

    pdists_mean_reduced = agg_func(np.array(individual_variable_dists))

    return pdists_mean_reduced

def hierarchical_distance_aggregate_score(X):
    """
    X: ndarray of features where the first dimension is ordered hierarchically (e.g. [Chromosome #, arm, region, band])
    """
    for i in range(1, len(X)):
        X[i][np.where(X[i-1] >= X[i])] = X[i-1][np.where(X[i-1] >= X[i])] # the distance of child feature is only as great as distance of parent features

    return np.nanmean(X, axis=0)


def seq_global_alignment_pairwise_score(u, v, truncate=True, min_length=1000):
    if isinstance(u[0], str) and isinstance(v[0], str):
        if ~truncate and (len(u[0]) > min_length or len(v[0]) > min_length):
            return np.nan
        return pairwise2.align.globalxx(u[0], v[0], score_only=True) / min(len(u[0]), len(v[0]))
    else:
        return np.nan
