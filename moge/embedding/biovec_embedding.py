from moge.embedding.static_graph_embedding import ImportedGraphEmbedding
import biovec
import numpy as np


class BioVecEmbedding(ImportedGraphEmbedding):
    def __init__(self, network, model_path_dict, d=100, method_name="BioVec"):
        super().__init__(d, method_name)

        models = {}
        for modality in model_path_dict:
            models[modality] = biovec.models.load_protvec(model_path_dict[modality])

        self.node_list = network.node_list
        self._X = np.zeros((len(self.node_list), d))

        for modality in model_path_dict:
            for node in network.nodes[modality]:
                if node not in self.node_list:
                    continue
                node_seq = network.genes_info.loc[node, "Transcript sequence"]

                if type(node_seq) is list:
                    node_seq = node_seq[0]
                elif node_seq is None:
                    continue

                node_emb = np.array(models[modality].to_vecs(node_seq)).sum(axis=0)
                self._X[self.node_list.index(node)] = node_emb
