import gseapy as gp
import numpy as np
import pandas as pd
from sklearn.metrics import homogeneity_score, completeness_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score

@DeprecationWarning
def evaluate_clustering(embedding, annotations, nodelist, node_label, n_clusters=None,
                        metrics=["homogeneity", "completeness", "nmi"], max_clusters=None, delim="\||;"):
    if annotations.loc[nodelist, node_label].dtypes == np.object \
            and annotations.loc[nodelist, node_label].str.contains(delim, regex=True).any():

        y_true = annotations.loc[nodelist, node_label].str.split(delim, expand=False)
        y_true = y_true.map(lambda x: sorted(x)[0] if x and len(x) >= 1 else None)
    else:
        y_true = annotations.loc[nodelist, node_label].map(
            lambda x: sorted(x)[0] if x and len(x) >= 1 else None)

    if n_clusters is None:
        n_clusters = min(len(y_true.unique()), max_clusters) if max_clusters else len(y_true.unique())
        print("Clustering", len(nodelist), "nodes with n_clusters:", n_clusters)
    try:
        y_pred = embedding.predict_cluster(n_clusters, node_list=nodelist)
    except AttributeError as e:
        return e

    return clustering_metrics(y_true, y_pred, metrics)


def clustering_metrics(y_true, y_pred, metrics):
    results = {}
    for metric in metrics:
        if metric == "homogeneity":
            results[metric] = homogeneity_score(y_true, y_pred)
        elif metric == "completeness":
            results[metric] = completeness_score(y_true, y_pred)
        elif metric == "nmi":
            results[metric] = normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic")
        elif metric == "ami":
            results[metric] = adjusted_mutual_info_score(y_true, y_pred)
    return results


def _get_top_enrichr_term(gene_sets, libraries=[
    # 'GO_Biological_Process_2018',
    # 'GO_Cellular_Component_2018',
    # 'GO_Molecular_Function_2018',
                                                'KEGG_2019_Human', ],
                          cutoff=0.01, top_k=1):
    results = []

    for gene_set in gene_sets:
        try:
            enr = gp.enrichr(gene_list=gene_set,
                             gene_sets=libraries,
                             cutoff=cutoff,
                             no_plot=True, verbose=False,
                             )
            if enr.results.shape[0] > 0:
                results.append(enr.results.sort_values(by="Adjusted P-value").head_node_type(top_k))
        except Exception:
            pass
    results = [row for row in results if row is not None]
    if len(results) > 0:
        return pd.concat(results)
    else:
        return None


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out
