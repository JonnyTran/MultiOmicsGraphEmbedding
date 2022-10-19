import pickle
from argparse import Namespace
from collections import defaultdict

import dgl
import numpy as np
import pandas as pd

from moge.dataset.PyG.hetero_generator import HeteroNodeClfDataset
from moge.dataset.sequences import SequenceTokenizers
from moge.dataset.utils import get_edge_index_dict
from moge.model.dgl.DeepGraphGO import load_protein_dataset
from moge.model.utils import tensor_sizes
from moge.network.hetero import HeteroNetwork
from openomics.database.ontology import UniProtGOA


def load_uniprotgoa(name: str, dataset_path: str, hparams: Namespace) -> HeteroNodeClfDataset:
    use_reverse = hparams.use_reverse
    head_node_type = hparams.head_node_type

    # Set ntypes to include
    hparams.ntype_subset = hparams.ntype_subset.split(" ") \
        if 'ntype_subset' in hparams and hparams.ntype_subset else None
    hparams.pred_ntypes = hparams.pred_ntypes.split(" ") \
        if 'pred_ntypes' in hparams and hparams.pred_ntypes else []

    # Set etypes to include
    if 'etype_subset' in hparams and hparams.etype_subset:
        hparams.etype_subset = hparams.etype_subset.split(" ")
    else:
        hparams.etype_subset = None

    geneontology = UniProtGOA(path='~/Bioinformatics_ExternalData/UniProtGOA/',
                              file_resources={"go.obo": "http://current.geneontology.org/ontology/go.obo",
                                              "goa_uniprot_all.processed.parquet": "goa_uniprot_all.processed.parquet"},
                              species=None)

    all_go = set(geneontology.network.nodes).intersection(geneontology.data.index)
    go_nodes = np.array(list(all_go))

    # Load HeteroNetwork from pickle
    with open(dataset_path, "rb") as file:
        network: HeteroNetwork = pickle.load(file)
        if not hasattr(network, 'train_nodes'):
            network.train_nodes = defaultdict(set)
            network.valid_nodes = defaultdict(set)
            network.test_nodes = defaultdict(set)

    # Add GO interactions
    network.add_edges_from_ontology(geneontology, nodes=go_nodes, split_ntype='namespace', etypes=hparams.etype_subset)

    for dst_ntype in set(['biological_process', 'molecular_function', 'cellular_component']).difference(
            hparams.pred_ntypes):
        train_date = '2018-01-01'
        valid_date = (pd.to_datetime(train_date) + pd.to_timedelta(26, "W")).date().strftime("%Y-%m-%d")
        test_date = '2021-04-01'
        network.add_edges_from_annotations(geneontology, filter_dst_nodes=network.nodes[dst_ntype],
                                           src_ntype=head_node_type, dst_ntype=dst_ntype,
                                           src_node_col='protein_id',
                                           train_date=train_date, valid_date=valid_date, test_date=test_date,
                                           use_neg_annotations=False)

    # Add GOA's from DeepGraphGO to UniProtGOA
    annot_df = network.annotations[head_node_type]
    annot_df['go_id'] = annot_df['go_id'].map(lambda d: d if isinstance(d, (list, np.ndarray)) else [])

    dgg_go_id = load_protein_dataset(hparams.deepgraphgo_data, namespaces=['mf', 'bp', 'cc'])
    annot_df = annot_df.join(dgg_go_id[['train_mask', 'valid_mask', 'test_mask']], on='protein_id')
    if 'DeepGraphGO' in name:
        dgg_go_id['go_id'] = dgg_go_id['go_id'].map(
            lambda d: d if isinstance(d, (list, np.ndarray)) else [])
        annot_df['go_id'] = annot_df['go_id'].map(list) + dgg_go_id.loc[annot_df.index, 'go_id']
        annot_df['go_id'] = annot_df['go_id'].map(np.unique).map(list)

    # Set train/valid/test split based on DeepGraphGO
    network.train_nodes[head_node_type] = set(annot_df.query('train_mask == True').index)
    network.valid_nodes[head_node_type] = set(annot_df.query('valid_mask == True').index)
    network.test_nodes[head_node_type] = set(annot_df.query('test_mask == True').index)
    network.set_edge_traintest_mask()

    # Set classes
    if isinstance(hparams.pred_ntypes, str):
        hparams.pred_ntypes = hparams.pred_ntypes.split(" ")
        go_classes = geneontology.data.index[geneontology.data['namespace'].isin(hparams.pred_ntypes)]
    else:
        raise Exception("Must provide `hparams.pred_ntypes` as a space-delimited string")

    # Neighbor loader
    if hparams.neighbor_loader == "NeighborLoader":
        hparams.neighbor_sizes = [8] * hparams.n_layers
    else:
        hparams.neighbor_sizes = [hparams.n_neighbors] * max(hparams.n_layers, hparams.t_order)

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

    # Create dataset
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
        pred_ntypes=hparams.pred_ntypes,
        ntype_subset=hparams.ntype_subset \
            if hparams.ntype_subset else set(network.nodes.keys()).difference([hparams.pred_ntypes]),
        exclude_etypes=[
            (head_node_type, 'associated', go_ntype) for go_ntype in hparams.pred_ntypes], )

    dataset._name = name
    print(dataset.G)

    # Create cls_graph at output layer
    if set(dataset.nodes.keys()).isdisjoint(hparams.pred_ntypes) and 'cls_graph' in hparams and hparams.cls_graph:
        cls_network_nodes = dataset.classes.tolist() + go_classes.difference(dataset.classes).tolist()
        cls_network = dgl.heterograph(
            get_edge_index_dict(geneontology.network, nodes=cls_network_nodes, format="dgl"))
        hparams.cls_graph = cls_network
    else:
        hparams.cls_graph = None

    print(pd.DataFrame(tensor_sizes(dict(
        train={ntype: dataset.G[ntype].train_mask.sum() for ntype in dataset.G.node_types if
               hasattr(dataset.G[ntype], 'train_mask')},
        valid={ntype: dataset.G[ntype].valid_mask.sum() for ntype in dataset.G.node_types if
               hasattr(dataset.G[ntype], 'valid_mask')},
        test={ntype: dataset.G[ntype].test_mask.sum() for ntype in dataset.G.node_types if
              hasattr(dataset.G[ntype], 'test_mask')}))).T)

    return dataset
