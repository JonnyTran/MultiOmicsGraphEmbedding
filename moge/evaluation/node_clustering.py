from sklearn.metrics import homogeneity_score, completeness_score, normalized_mutual_info_score


def evaluate_clustering(embedding, network, node_label="locus_type", n_clusters=None,
                        metrics=["homogeneity", "completeness", "nmi"]):
    nodelist = embedding.node_list
    genes_info = network.genes_info
    nodes_with_label = genes_info[genes_info[node_label].notna()].index
    nodelist = [node for node in nodelist if node in nodes_with_label]

    y_true = genes_info.loc[nodelist, node_label]

    if n_clusters is None:
        n_clusters = len(y_true.unique())

    y_pred = embedding.predict_cluster(n_clusters, node_list=nodelist)
    assert len(y_pred) == len(y_true)

    results = {}
    for metric in metrics:
        if metric == "homogeneity":
            results[metric] = homogeneity_score(y_true, y_pred)
        elif metric == "completeness":
            results[metric] = completeness_score(y_true, y_pred)
        elif metric == "nmi":
            results[metric] = normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic")

    return results
