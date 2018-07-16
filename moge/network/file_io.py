import os

import numpy as np
import pandas as pd
import networkx as nx

from TCGAMultiOmics.multiomics import MultiOmicsData
from moge.network.heterogeneous_network import HeterogeneousNetwork
from sklearn.feature_extraction import DictVectorizer


def write_node_labels_to_file(file_path, multi_omics_data:MultiOmicsData, modalities=["GE", "MIR", "LNC"], label_cols=["Disease association"], sep="\t", remove_na=True, get_dummies=False):
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
            elif "Disease association" in label_cols:

                genes_info_concat.append(multi_omics_data[modality].get_genes_info()[label_cols])


        if remove_na:
            entries_to_write = pd.concat(genes_info_concat, axis=0)[label_cols[0]].dropna()
        else:
            entries_to_write = pd.concat(genes_info_concat, axis=0)[label_cols[0]]

        if get_dummies:
            entries_to_write = entries_to_write.str.get_dummies("|")

        file.write(entries_to_write.to_csv(sep=sep, header=None, index_label=True))

        file.close()
