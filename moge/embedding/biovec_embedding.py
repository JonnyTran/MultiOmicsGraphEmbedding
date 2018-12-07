import biovec
import numpy as np

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
