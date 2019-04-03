import biovec
import numpy as np
import pandas as pd

from moge.embedding.static_graph_embedding import ImportedGraphEmbedding


class BioVecEmbedding(ImportedGraphEmbedding):
    def __init__(self, network, model_path_dict, d=100, method_name="BioVec"):
        super().__init__(d, method_name)

        models = {}
        for modality in model_path_dict:
            models[modality] = biovec.models.load_protvec(model_path_dict[modality])

        self.node_list = []
        self._X = []

        for node in network.node_list:
            modality = network.node_to_modality[node]
            node_seq = network.genes_info.loc[node, "Transcript sequence"]
            if type(node_seq) is list:
                node_seq = node_seq[0]
            elif node_seq is None:
                continue

            try:
                node_emb = np.array(models[modality].to_vecs(node_seq)).sum(axis=0)
            except Exception:
                print("Failed to get vectors for", node)
                continue

            self.node_list.append(node)
            self._X.append(node_emb)

        assert len(self._X) == len(self.node_list)
        self._X = np.array(self._X)

class iDeepVEmbedding(ImportedGraphEmbedding):
    def __init__(self, network, model_path_dict, d=100, method_name="iDeepV"):
        super().__init__(d, method_name)


class LncTarInteraction(ImportedGraphEmbedding):
    def __init__(self, table_file):
        self.table = pd.read_table(table_file)
        # self.edges_pred = self.table[["Query", "Target"]]
        self.node_list = list(set(self.table["Query"].unique()) | set(self.table["Target"].unique()))
        print("node_list size", len(self.node_list))

    def get_top_k_predicted_edges(self, edge_type=None, top_k=100, node_list=None, node_list_B=None,
                                  training_network=None, databases=None):
        table = self.table[self.table["Query"].isin(node_list) & self.table["Target"].isin(node_list)]

        return table.sort_values(by="ndG").loc[:top_k, ["Query", "Target", "ndG"]].values.tolist()
