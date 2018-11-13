from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn import svm


def evaluate_classification(embedding, network, node_label="Family", cv=5,
                            scoring=['precision_macro', 'recall_macro']):
    nodelist = embedding.node_list
    genes_info = network.genes_info
    nodes_with_label = genes_info[genes_info[node_label].notna()].index
    nodelist = [node for node in nodelist if node in nodes_with_label]
    node_groups = genes_info.loc[nodelist, "locus_type"]

    X = embedding.get_embedding()
    y = genes_info.loc[nodelist, node_label]

    clf = svm.LinearSVC(multi_class="crammer_singer")
    scores = cross_validate(clf, X, y, groups=node_groups, cv=cv, n_jobs=-2, scoring=scoring)

    return scores
