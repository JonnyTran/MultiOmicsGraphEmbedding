import os

import numpy as np
import pandas as pd
import networkx as nx

from TCGAMultiOmics.multiomics import MultiOmicsData
from moge.network.heterogeneous_network import HeterogeneousNetwork
from sklearn.feature_extraction import DictVectorizer


def write_node_labels_to_file(file_path, network:HeterogeneousNetwork, multi_omics_data:MultiOmicsData, number_of_labels=None, node_label_integer=False, dummyize=False):
    with open(file_path, 'a') as file:
        for modality in network.modalities:

            if modality == "GE":
                label_cols = ["gene_family_id"]
            elif modality == "MIR":
                label_cols = ["miR family"]
            elif modality == "LNC":
                label_cols = ["Transcript Type"]
            else:
                raise Exception("Modality not supported. Does not know which label information to get from genes info for modality" + modality)

            genes_info = multi_omics_data[modality].get_genes_info()

            file.write(genes_info.to_csv(sep="\t", header=None, index_label=True, line_terminator="\n", columns=label_cols))

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