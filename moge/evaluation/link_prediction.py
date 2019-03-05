import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
from sklearn.metrics import average_precision_score ,precision_recall_curve
from moge.evaluation.metrics import link_prediction_score
from moge.evaluation.utils import mask_test_edges, split_train_test_edges
from moge.network.heterogeneous_network import HeterogeneousNetwork


def evaluate_pr_curve_link_pred_by_database(methods, data_generator,
                                            databases=["lncrna2target_high", "miRTarBase", "BioGRID", "lncRInter"]):
    for database in databases:
        print(database)
        data_generator.reload_directed_edges_data(edge_types=["d"], databases=[database, ])
        X, y_true = data_generator.make_dataset()
        y_true = y_true.astype(int)
        print("y_true.shape", y_true.shape)
        evaluate_pr_curve_link_pred(methods, X, y_true, title=database + "PR curve")


def evaluate_pr_curve_link_pred(methods, X, y_true, title='PR curve', dpi=300, fig_save_path=None):
    fig = plt.figure(figsize=(4, 4), dpi=dpi)
    ax = fig.add_subplot(111)

    color_dict = {"LINE":"b", "HOPE":"b", "SDNE":"y", "node2vec":"g", "rna2rna":"r", "siamese":"r"}
    ls_dict = {"LINE":":", "HOPE":"-", "SDNE":"--", "node2vec":"--", "rna2rna":"-", "siamese":":"}

    for method in methods.keys():
        print("Method:", method)
        y_prob_pred = methods[method].predict(X)
        average_precision = average_precision_score(y_true=y_true, y_score=y_prob_pred)
        precision, recall, _ = precision_recall_curve(y_true=y_true, probas_pred=y_prob_pred, pos_label=1)

        ax.plot(recall, precision, color=color_dict[method], ls=ls_dict[method],
                label=method + '. AUPR={0:0.2f}'.format(average_precision))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.00])
    plt.xlim([0.0, 1.0])
    plt.legend(loc="lower right")
    plt.title(title)
    plt.show()
    if fig_save_path is not None:
        fig.savefig(fig_save_path, bbox_inches='tight')


def evaluate_top_k_link_pred(embedding, network_train, network_test, node_list, edge_type="d", top_k=100):
    nodes = embedding.node_list
    nodes = [node for node in nodes if node in node_list]

    edges_pred = embedding.get_top_k_predicted_edges(edge_type=edge_type, top_k=top_k,
                                                     node_list=nodes,
                                                     training_network=network_train)
    edges_true = network_test.get_edgelist(edge_types=[edge_type], node_list=nodes, inclusive=True)

    results = {}
    results["precision"] = precision(edges_true, edges_pred)
    results["recall"] = recall(edges_true, edges_pred)
    return results


def precision(edges_true, edges_pred):
    if len(edges_pred[0]) > 2:
        edges_pred = [(u, v) for u, v, _ in edges_pred]

    true_positives = len(set(edges_true) & set(edges_pred))
    return true_positives / len(edges_pred)


def recall(edges_true, edges_pred):
    if len(edges_pred[0]) > 2:
        edges_pred = [(u, v) for u, v, _ in edges_pred]

    true_positives = len(set(edges_true) & set(edges_pred))
    return true_positives / len(edges_true)



def select_random_link_predictions(top_k, estimated_adj, excluding_edges, seed=0):
    np.random.seed(seed)
    random_adj = np.random.rand(*estimated_adj.shape)
    random_adj[excluding_edges[:, 0], excluding_edges[:, 1]] = 0

    top_k_indices = largest_indices(random_adj, top_k)

    return top_k_indices


def largest_indices(array, k, smallest=False):
    """Returns the k largest indices from a numpy array using partition O(n + k lg k) """
    flat = array.flatten()
    indices = np.argpartition(flat, -k)[-k:]
    if smallest:
        indices = indices[np.argsort(flat[indices])]
    else:
        indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, array.shape)













