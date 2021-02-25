import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from moge.evaluation.utils import largest_indices


def evaluate_pr_curve_link_pred_by_database(methods, data_generator, edge_types=["d"],
                                            tests=[("LNC", "GE", "lncrna2target", "LncRNA2Target"),
                                                   ("MIR", "GE", "miRTarBase", "miRTarBase"),
                                                   ("GE", "GE", "BioGRID", "BioGRID"),
                                                   ("LNC", "GE", "NPInter", "NPInter")]):
    for source, target, database, title in tests:
        print(database)
        total_edges = data_generator.reload_directed_edges_data(edge_types=edge_types,
                                                                databases=database if type(database) == list else [
                                                                    database, ],
                                                                node_list=data_generator.network.nodes[
                                                      source] if source is not None else None,
                                                                node_list_B=data_generator.network.nodes[
                                                      target] if target is not None else None)
        if total_edges > 0:
            data_generator.on_epoch_end()
        else:
            continue

        X, y_true = data_generator.load_data()
        y_true = np.round(y_true)
        evaluate_pr_curve_link_pred(methods, X, y_true, title=title, data_generator=data_generator)


def evaluate_pr_curve_link_pred(methods, X, y_true, title='PR curve', dpi=200, fig_save_path=None, data_generator=None):
    fig = plt.figure(figsize=(4, 4), dpi=dpi)
    ax = fig.add_subplot(111)

    color_dict = {"LINE": "b", "HOPE": "c", "SDNE": "y", "node2vec": "g", "rna2rna": "r", "siamese": "r"}
    ls_dict = {"LINE": ":", "HOPE": "-.", "SDNE": "--", "node2vec": "--", "rna2rna": "-", "siamese": ":"}

    for method in methods.keys():
        if method == "BioVec": continue
        # if method is "siamese" and method == list(methods.keys())[-1]:
        #     y_prob_pred = methods[method].predict_generator(data_generator)
        # else:
        y_prob_pred = methods[method].predict(X,,
                      average_precision = average_precision_score(y_true=y_true, y_score=y_prob_pred)
        precision, recall, _ = precision_recall_curve(y_true=y_true, probas_pred=y_prob_pred, pos_label=1)

        ax.plot(recall, precision, color=color_dict[method], ls=ls_dict[method],
                label=method + '. AUPR={0:0.2f}'.format(average_precision))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.00])
    plt.ylim([0.0, 1.00])
    plt.legend(loc="lower right")
    plt.title(title)
    plt.show()
    if fig_save_path is not None:
        fig.savefig(fig_save_path, bbox_inches='tight')


def evaluate_top_k_link_pred(embedding, network_train, network_test, node_list, node_list_B=None, edge_type=None,
                             top_k=100, databases=None):
    if node_list_B is not None:
        nodes = [n for n in embedding.node_list if n in node_list or n in node_list_B]
        nodes_A = [n for n in embedding.node_list if n in node_list]
        nodes_B = [n for n in embedding.node_list if n in node_list_B]
    else:
        nodes = [node for node in embedding.node_list if node in node_list]

    if node_list_B is not None:
        edges_pred = embedding.get_top_k_predicted_edges(edge_type=edge_type, top_k=top_k,
                                                         node_list=nodes_A, node_list_B=nodes_B,
                                                         training_network=network_train, databases=databases)
    else:
        edges_pred = embedding.get_top_k_predicted_edges(edge_type=edge_type, top_k=top_k,
                                                         node_list=nodes,
                                                         training_network=network_train, databases=databases)
    edges_true = network_test.get_edgelist(node_list=nodes, inclusive=True, databases=databases)

    if len(edges_true) == 0:
        print("no edges in edges_true")
        return

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


