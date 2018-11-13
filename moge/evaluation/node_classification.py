from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from sklearn import svm


def evaluate_classification(embedding, network, node_label="Family", cv=5, multilabel=False,
                            scoring=['precision_macro', 'recall_macro']):
    nodelist = embedding.node_list
    genes_info = network.genes_info
    nodes_with_label = genes_info[genes_info[node_label].notna()].index
    nodelist = [node for node in nodelist if node in nodes_with_label]
    nodes_split_by_group = genes_info.loc[nodelist, "locus_type"]

    X = embedding.get_embedding()
    X = X[nodelist, :]
    if multilabel:
        labels = genes_info.loc[nodelist, node_label].str.split("|", expand=False)
        labeler = MultiLabelBinarizer()
        y = labeler.fit_transform(labels.tolist())
        clf = KNeighborsClassifier(n_neighbors=10, weights="distance", algorithm="auto", metric="euclidean")

    else:  # Multiclass classification (only single label each sample)
        y = genes_info.loc[nodelist, node_label].str.split("|", expand=True)[0]
        clf = svm.LinearSVC(multi_class="ovr")

    scores = cross_validate(clf, X, y, groups=nodes_split_by_group, cv=cv, n_jobs=-2, scoring=scoring)

    return scores
