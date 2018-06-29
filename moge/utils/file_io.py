import os

import numpy as np
import pandas as pd
import networkx as nx

from TCGAMultiOmics.multiomics import MultiOmicsData
from moge.network.heterogeneous_network import HeterogeneousNetwork
from sklearn.feature_extraction import DictVectorizer


def write_node_labels_to_file(file_path, multi_omics_data:MultiOmicsData, modalities=["GE", "MIR", "LNC"], label_cols=["Disease association"],sep=" ", get_dummies=True):
    with open(file_path, 'w') as file:
        genes_info_concat = []

        for modality in modalities:
            if label_cols == ["family"]:
                if modality == "GE":
                    family_col = ["gene_family_id"]
                elif modality == "MIR":
                    family_col = ["miR family"]
                elif modality == "LNC":
                    family_col = ["Transcript Type"]

                genes_info_concat.append(multi_omics_data[modality].get_genes_info().rename(columns={family_col[0]: "family"})[label_cols])
            else:
                genes_info_concat.append(multi_omics_data[modality].get_genes_info()[label_cols])

        if get_dummies:
            file.write(pd.concat(genes_info_concat, axis=0)[label_cols[0]].str.get_dummies("|").to_csv(sep=sep, header=None, index_label=True))
        else:
            file.write(pd.concat(genes_info_concat, axis=0)[label_cols[0]].to_csv(sep=sep, header=None, index_label=True))

#             for node in network.nodes[modality]:
#                 gene_info = multi_omics_data[modality].get_genes_info()[node]
#                 # TODO if node_label_integer:
#                 file.write(node)
#
#                 # str = gene_info.loc[node, label_cols].to_csv(sep="\t")
#
# )
#
#                 file.write(os.linesep)
        file.close()


def write_node_features_to_file(file_path, network:HeterogeneousNetwork, multi_omics_data:MultiOmicsData):
    pass

def import_graph_from_files(node_features_path, edgelist_path, node_labels_path):
    pass