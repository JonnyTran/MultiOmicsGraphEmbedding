import pickle

import dgl
import numpy as np
import pandas as pd
from openomics.database.ontology import UniProtGOA

from moge.dataset.PyG.hetero_generator import HeteroNodeClfDataset
from moge.dataset.sequences import SequenceTokenizers
from moge.dataset.utils import get_edge_index_dict
from moge.model.dgl.DeepGraphGO import load_protein_dataset
from moge.model.utils import tensor_sizes


def load_uniprotgoa(dataset_path, head_node_type, hparams, name, use_reverse):
    with open(dataset_path, "rb") as file:
        network = pickle.load(file)
        if not hasattr(network, 'train_nodes'):
            network.train_nodes, network.valid_nodes, network.test_nodes = {}, {}, {}
    geneontology = UniProtGOA(file_resources={"go.obo": "http://current.geneontology.org/ontology/go.obo", })
    annot_df = network.annotations[head_node_type]
    annot_df['go_id'] = annot_df['go_id'].map(lambda d: d if isinstance(d, (list, np.ndarray)) else [])
    # Add GO interactions
    all_go = set(geneontology.network.nodes).intersection(geneontology.data.index)
    go_nodes = np.array(list(all_go))
    network.add_edges_from_ontology(geneontology, nodes=go_nodes, etypes=["is_a", "part_of"])
    # Add GOA's from DeepGraphGO to UniProtGOA
    uniprot_go_id = load_protein_dataset(hparams.deepgraphgo_data, namespaces=['mf', 'bp', 'cc'])
    annot_df = annot_df.join(uniprot_go_id[['train_mask', 'valid_mask', 'test_mask']], on='protein_id')
    if 'DeepGraphGO' in name:
        uniprot_go_id['go_id'] = uniprot_go_id['go_id'].map(
            lambda d: d if isinstance(d, (list, np.ndarray)) else [])
        annot_df['go_id'] = annot_df['go_id'] + uniprot_go_id.loc[annot_df.index, 'go_id']
        annot_df['go_id'] = annot_df['go_id'].map(np.unique).map(list)
    # Set train/valid/test split based on DeepGraphGO
    network.train_nodes[head_node_type] = set(annot_df.query('train_mask == True').index)
    network.valid_nodes[head_node_type] = set(annot_df.query('valid_mask == True').index)
    network.test_nodes[head_node_type] = set(annot_df.query('test_mask == True').index)
    network.set_edge_traintest_mask(network.train_nodes, network.valid_nodes, network.test_nodes,
                                    exclude_metapaths=[] if not hparams.inductive else None)
    # Set classes
    namespaces = set(hparams.namespaces) if not isinstance(hparams.namespaces,
                                                           (list, set, tuple)) else hparams.namespaces
    go_classes = geneontology.data.index[geneontology.data['namespace'].str.get(0).isin(namespaces)]
    hparams.namespaces = geneontology.data.loc[go_classes, 'namespace'].unique().tolist()
    # Neighbor loader
    if hparams.neighbor_loader == "HGTLoader":
        hparams.neighbor_sizes = [hparams.n_neighbors // 8] * hparams.n_layers
    else:
        hparams.neighbor_sizes = [hparams.n_neighbors] * hparams.n_layers
    # Sequences
    if hasattr(hparams, 'sequence') and hparams.sequence:
        sequence_tokenizers = SequenceTokenizers(
            vocabularies={"MicroRNA": "armheb/DNA_bert_3",
                          "LncRNA": "armheb/DNA_bert_6",
                          "MessengerRNA": "armheb/DNA_bert_6",
                          'Protein': 'zjukg/OntoProtein',
                          'GO_term': "dmis-lab/biobert-base-cased-v1.2", },
            max_length=hparams.max_length)
    else:
        sequence_tokenizers = None
    hparams.ntype_subset = hparams.ntype_subset.split(" ")
    print('hparams.ntype_subset', hparams.ntype_subset)
    dataset = HeteroNodeClfDataset.from_heteronetwork(
        network, target="go_id",
        labels_subset=geneontology.data.index.intersection(go_classes),
        min_count=50,
        expression=hparams.feature if 'feature' in hparams else True,
        sequence=True if sequence_tokenizers is not None else False,
        add_reverse_metapaths=use_reverse,
        head_node_type=hparams.head_node_type,
        neighbor_loader=hparams.neighbor_loader,
        neighbor_sizes=hparams.neighbor_sizes,
        split_namespace=True,
        go_ntype="GO_term",
        seq_tokenizer=sequence_tokenizers,
        inductive=hparams.inductive,
        ntype_subset=hparams.ntype_subset \
            if hparams.ntype_subset else set(network.nodes.keys()).difference(['GO_term']),
        exclude_metapaths=[
            (head_node_type, 'associated', 'GO_term'),
            ('GO_term', 'rev_associated', head_node_type),
            (head_node_type, 'associated', 'BPO'),
            ('BPO', 'rev_associated', head_node_type),
            (head_node_type, 'associated', 'MFO'),
            ('MFO', 'rev_associated', head_node_type),
            (head_node_type, 'associated', 'CCO'),
            ('CCO', 'rev_associated', head_node_type)
        ])
    dataset._name = name
    print(dataset.G)
    if hasattr(hparams, 'cls_graph') and hparams.cls_graph:
        cls_network_nodes = dataset.classes.tolist() + go_classes.difference(dataset.classes).tolist()
        cls_network = dgl.heterograph(
            get_edge_index_dict(geneontology.network, nodes=cls_network_nodes, format="dgl"))
        hparams.cls_graph = cls_network
    print(pd.DataFrame(tensor_sizes(dict(
        train={ntype: dataset.G[ntype].train_mask.sum() for ntype in dataset.G.node_types if
               hasattr(dataset.G[ntype], 'train_mask')},
        valid={ntype: dataset.G[ntype].valid_mask.sum() for ntype in dataset.G.node_types if
               hasattr(dataset.G[ntype], 'valid_mask')},
        test={ntype: dataset.G[ntype].test_mask.sum() for ntype in dataset.G.node_types if
              hasattr(dataset.G[ntype], 'test_mask')}))).T)
    return dataset
