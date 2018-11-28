from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MultiLabelBinarizer


def evaluate_classification(embedding, network, cv=5, node_label="Family", multilabel=False,
                            scoring=['precision_micro', 'recall_micro', "f1_micro"], verbose=False):
    nodelist = embedding.node_list
    genes_info = network.genes_info

    # Filter labels that have enough samples
    if multilabel:
        labels_to_test = [i for (i, v) in (genes_info[genes_info[node_label].notnull()][node_label].str.get_dummies(
            "|").sum(axis=0) >= cv).items() if v == True]
        nodes_with_label = [i for (i, v) in genes_info[genes_info[node_label].notnull()][node_label].str.contains(
            "|".join(labels_to_test)).items() if v == True]
    else:
        labels_to_test = [i for (i, v) in (genes_info[node_label].value_counts() >= cv).items() if v == True]
        nodes_with_label = genes_info[genes_info[node_label].isin(labels_to_test)].index

    nodelist = [node for node in nodelist if node in nodes_with_label]
    nodes_split_by_group = genes_info.loc[nodelist, node_label].str.split("|", expand=True)[0]
    print("labels_to_test", labels_to_test) if verbose > 1 else None
    print("# of labels with >cv samples:", len(labels_to_test), ", # of nodes to train/test:", len(nodelist)) if verbose else None

    X = embedding.get_embedding(node_list=nodelist)
    assert len(nodelist) == X.shape[0]

    if multilabel:
        labels = genes_info.loc[nodelist, node_label].str.split("|", expand=False)
        y = MultiLabelBinarizer().fit_transform(labels.tolist())
        # clf = KNeighborsClassifier(n_neighbors=10, weights="distance", algorithm="auto", metric="euclidean")
        clf = RandomForestClassifier(n_estimators=10, n_jobs=-2)

    else:  # Multiclass classification (only single label each sample)
        y = genes_info.loc[nodelist, node_label].str.split("|", expand=True)[0]
        clf = svm.LinearSVC(multi_class="ovr")

    scores = cross_validate(clf, X, y, groups=nodes_split_by_group, cv=cv, n_jobs=-2, scoring=scoring,
                            return_train_score=False)

    return scores
