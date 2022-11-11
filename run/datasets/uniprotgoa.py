import os.path
import pickle
import traceback
from argparse import Namespace
from collections import defaultdict
from collections.abc import Iterable
from os.path import join
from typing import List, Mapping

import dask.dataframe as dd
import joblib
import networkx as nx
import numpy as np
import pandas as pd
from logzero import logger
from sklearn.preprocessing import MultiLabelBinarizer
from torch_geometric.utils import to_undirected

from moge.dataset.PyG.hetero_generator import HeteroNodeClfDataset
from moge.dataset.sequences import SequenceTokenizers
from moge.model.dgl.DeepGraphGO import load_protein_dataset
from moge.network.hetero import HeteroNetwork
from moge.network.utils import to_list_of_strs
from openomics.database.ontology import UniProtGOA, get_predecessor_terms


def get_load_path(name, hparams: Namespace, labels_dataset, ntype_subset, pred_ntypes, add_parents, go_etypes,
                  exclude_etypes, feature, save_path):
    node_types = ['MicroRNA', 'MessengerRNA', 'LncRNA', 'Protein', 'biological_process', 'molecular_function',
                  'cellular_component']
    if ntype_subset:
        node_types = [ntype for ntype in node_types if ntype in ntype_subset]

    ntypes = ''.join(s.capitalize()[0] for s in node_types)
    options = f"{labels_dataset}.{'parents' if add_parents else 'child'}." \
              f"{'feature' if feature else 'nofeature'}"
    if 'species' in hparams:
        options = f"{hparams.species}.{options}"

    metapaths = ''.join([e[0] for e in go_etypes] if go_etypes else [])
    ex_etypes = ''.join(e.lower()[0] for s, e, d in exclude_etypes)
    pntypes = ''.join(''.join(s[0] for s in ntype.split("_")) for ntype in pred_ntypes)

    slug = f'{ntypes}-{metapaths}{ex_etypes}-{pntypes}'
    load_path = join(save_path, '.'.join([name, options, slug]))
    load_path = os.path.expanduser(load_path)

    return load_path


def build_uniprot_dataset(name: str, dataset_path: str, hparams: Namespace,
                          save_path='~/Bioinformatics_ExternalData/LATTE2GO/', save=True) \
        -> HeteroNodeClfDataset:
    target = 'go_id'

    add_parents, deepgraphgo_path, exclude_etypes, feature, go_etypes, head_ntype, labels_dataset, ntype_subset, \
    pred_ntypes, uniprotgoa_path, use_reverse = parse_options(hparams, dataset_path)

    load_path = get_load_path(name, hparams, labels_dataset, ntype_subset, pred_ntypes, add_parents, go_etypes,
                              exclude_etypes, feature, save_path)

    if os.path.exists(os.path.expanduser(join(load_path, "metadata.json"))):
        try:
            logger.info(f'Loading saved HeteroNodeClfDataset at {load_path}')
            dataset = HeteroNodeClfDataset.load(load_path, **hparams.__dict__)

            return dataset
        except Exception as e:
            traceback.print_exc()
            pass
    else:
        print("Building", os.path.basename(load_path))

    # Load UniProtGOA ontology
    geneontology = UniProtGOA(path=os.path.dirname(uniprotgoa_path),
                              file_resources={"go.obo": "http://current.geneontology.org/ontology/go.obo",
                                              "goa_uniprot_all.processed.parquet": uniprotgoa_path},
                              species=None,
                              blocksize=None)

    all_go = set(geneontology.network.nodes).intersection(geneontology.data.index)
    go_nodes = np.array(list(all_go))

    # Load HeteroNetwork
    if isinstance(dataset_path, str):
        with open(os.path.expanduser(dataset_path), "rb") as file:
            network: HeteroNetwork = pickle.load(file)
            if not hasattr(network, 'train_nodes'):
                network.train_nodes = defaultdict(set)
                network.valid_nodes = defaultdict(set)
                network.test_nodes = defaultdict(set)
    elif isinstance(dataset_path, HeteroNetwork):
        network = dataset_path

    else:
        raise Exception()

    # Replace alt_id for go_id labels
    alt_id2go_id = geneontology.get_mapper('alt_id', target).to_dict()

    if not ntype_subset or {'biological_process', 'cellular_component', 'molecular_function'}.intersection(
            ntype_subset):
        # Add GO ontology interactions
        if go_etypes:
            network.add_edges_from_ontology(geneontology, nodes=go_nodes, split_ntype='namespace', etypes=go_etypes)

        to_replace = geneontology.annotations['go_id'].isin(alt_id2go_id)
        geneontology.annotations.loc[to_replace, 'go_id'] = geneontology.annotations.loc[to_replace, 'go_id'].replace(
            alt_id2go_id)

        # Add Protein-GO annotations
        for dst_ntype in {'biological_process', 'molecular_function', 'cellular_component'}.difference(pred_ntypes):
            network.add_edges_from_annotations(geneontology, filter_dst_nodes=network.nodes[dst_ntype],
                                               src_ntype=head_ntype, dst_ntype=dst_ntype,
                                               src_node_col='protein_id',
                                               train_date=hparams.train_date,
                                               valid_date=hparams.valid_date,
                                               test_date=hparams.test_date,
                                               use_neg_annotations=False)
            # TODO whether to add annotation edges with parents

    # Set the go_id label and train/valid/test node split for head_node_type
    if labels_dataset.startswith('DGG'):
        network.annotations[head_ntype], train_nodes, valid_nodes, test_nodes = get_DeepGraphGO_split(
            network.annotations[head_ntype], deepgraphgo_path, target=target, pred_ntypes=pred_ntypes)

        network.train_nodes[head_ntype] = train_nodes
        network.valid_nodes[head_ntype] = valid_nodes
        network.test_nodes[head_ntype] = test_nodes


    elif labels_dataset.startswith("GOA"):
        index_name = network.annotations[head_ntype].index.name
        network.multiomics[head_ntype].annotate_attributes(geneontology.annotations, on=index_name,
                                                           columns=[target], agg='unique')

        protein_earliest = geneontology.annotations \
            .query(f'namespace in {pred_ntypes}') \
            .groupby(geneontology.annotations.index.name)['Date'].min()

        if isinstance(protein_earliest, dd.Series):
            protein_earliest = protein_earliest.compute()
        network.train_nodes[head_ntype] = set(protein_earliest.index[protein_earliest <= hparams.train_date])
        network.valid_nodes[head_ntype] = set(protein_earliest.index[(protein_earliest > hparams.train_date) &
                                                                     (protein_earliest <= hparams.valid_date)])
        network.test_nodes[head_ntype] = set(protein_earliest.index[(protein_earliest > hparams.valid_date) &
                                                                    (protein_earliest <= hparams.test_date)])

    else:
        raise Exception('`labels_dataset` must be "DGG" or "GOA"')

    # Classes (DGG)
    go_classes = []
    rename_dict = {'molecular_function': 'mf', 'biological_process': 'bp', 'cellular_component': 'cc',
                   'mf': 'mf', 'bp': 'bp', 'cc': 'cc'}
    for pred_ntype in pred_ntypes:
        namespace = rename_dict[pred_ntype]
        mlb: MultiLabelBinarizer = joblib.load(os.path.join(deepgraphgo_path, f'{namespace}_go.mlb'))
        go_classes.append(mlb.classes_)

    go_classes = np.hstack(go_classes) if len(go_classes) > 1 else go_classes[0]
    min_count = None

    logger.info(f'Loaded MultiLabelBinarizer with {go_classes.shape} {",".join(pred_ntypes)} classes')

    # Replace alt_id labels
    network.annotations[head_ntype][target].map(lambda li: _replace_alt_id(li, alt_id2go_id))

    # add parent terms to ['go_id'] column
    if add_parents:
        logger.info(f"add_parents, before: {network.annotations[head_ntype][target].dropna().map(len).mean()}")
        subgraph = geneontology.get_subgraph(edge_types="is_a")
        node_ancestors = {node: nx.ancestors(subgraph, node) for node in subgraph.nodes}
        agg = lambda s: get_predecessor_terms(s, g=node_ancestors, join_groups=True, keep_terms=True)

        network.annotations[head_ntype][target] = network.annotations[head_ntype][target].apply(agg)
        logger.info(f"add_parents, after: {network.annotations[head_ntype][target].dropna().map(len).mean()}")

    else:
        network.annotations[head_ntype][target] = network.annotations[head_ntype][target] \
            .fillna('').map(list).map(np.unique)

    # if hparams.inductive:
    #     network.set_edge_traintest_mask()

    # Neighbor loader
    max_order = max(hparams.n_layers, hparams.t_order if 't_order' in hparams else 0)

    if hparams.neighbor_loader == "NeighborLoader":
        hparams.neighbor_sizes = [8] * max_order
    else:
        hparams.neighbor_sizes = [hparams.n_neighbors] * max_order

    # Sequences
    if hasattr(hparams, 'sequence') and hparams.sequence:
        if 'vocabularies' in hparams:
            vocabularies = hparams.vocabularies
        else:
            vocabularies = {"MicroRNA": "armheb/DNA_bert_3",
                            "LncRNA": "armheb/DNA_bert_6",
                            "MessengerRNA": "armheb/DNA_bert_6",
                            'Protein': 'zjukg/OntoProtein',
                            'GO_term': "dmis-lab/biobert-base-cased-v1.2", }
        sequence_tokenizers = SequenceTokenizers(
            vocabularies=vocabularies,
            max_length=hparams.max_length if 'max_length' in hparams else None)
    else:
        sequence_tokenizers = None

    # Create dataset
    dataset = HeteroNodeClfDataset.from_heteronetwork(
        network,
        target=target,
        labels_subset=geneontology.data.index.intersection(go_classes),
        min_count=min_count,
        expression=feature,
        sequence=True if sequence_tokenizers is not None else False,
        seq_tokenizer=sequence_tokenizers,
        add_reverse_metapaths=use_reverse,
        head_node_type=head_ntype,
        neighbor_loader=hparams.neighbor_loader,
        neighbor_sizes=hparams.neighbor_sizes,
        split_namespace=True,
        inductive=hparams.inductive,
        pred_ntypes=pred_ntypes,
        ntype_subset=ntype_subset,
        exclude_etypes=exclude_etypes, )

    # Remove the reverse etype for undirected edge type
    if use_reverse and ('Protein', 'rev_protein-protein', 'Protein') in dataset.G.edge_types:
        del dataset.G[('Protein', 'rev_protein-protein', 'Protein')]
        dataset.G[('Protein', 'protein-protein', 'Protein')].edge_index = to_undirected(
            dataset.G[('Protein', 'protein-protein', 'Protein')].edge_index)
        dataset.metapaths.pop(dataset.metapaths.index(('Protein', 'rev_protein-protein', 'Protein')))

    # Post-processing
    for ntype in pred_ntypes:
        if ntype not in dataset.nodes_namespace:
            dataset.nodes_namespace[ntype] = geneontology.data.query(f'namespace == "{ntype}"')['namespace']
    dataset.nodes_namespace = {ntype: series.replace({'biological_process': 'BPO',
                                                      'cellular_component': 'CCO',
                                                      'molecular_function': 'MFO'}) \
                               for ntype, series in dataset.nodes_namespace.items()}
    for ntype, series in dataset.nodes_namespace.items():
        series.fillna(series.mode()[0], inplace=True)

    dataset.nodes_namespace[dataset.head_node_type] = dataset.network.annotations[dataset.head_node_type]["species_id"]

    # Save hparams attr
    hparams.min_count = min_count
    hparams.n_classes = dataset.n_classes
    dataset.hparams = hparams
    dataset._name = os.path.basename(load_path)

    if save:
        dataset.save(load_path, add_slug=False)

    return dataset

def parse_options(hparams, dataset_path):
    # Set arguments
    use_reverse = hparams.use_reverse
    head_ntype = hparams.head_node_type
    add_parents = hparams.add_parents
    feature = hparams.feature
    assert isinstance(feature, bool)
    uniprotgoa_path = hparams.uniprotgoa_path
    deepgraphgo_path = hparams.deepgraphgo_data
    labels_dataset = hparams.labels_dataset

    # Species-specific dataset
    if isinstance(dataset_path, str) and 'HUMAN_MOUSE' in dataset_path:
        hparams.species = 'HUMAN_MOUSE'
    elif isinstance(dataset_path, str) and 'HUMAN' in dataset_path:
        hparams.species = 'HUMAN'
    elif isinstance(dataset_path, str) and not 'HUMAN' in dataset_path:
        hparams.species = 'MULTISPECIES'

    # ntype_subset
    if 'ntype_subset' in hparams and isinstance(hparams.ntype_subset, str):
        if len(hparams.ntype_subset) == 0:
            ntype_subset = [head_ntype]
        else:
            ntype_subset = hparams.ntype_subset.split(" ")
    elif 'ntype_subset' in hparams and isinstance(hparams.ntype_subset, Iterable):
        ntype_subset = hparams.ntype_subset
    else:
        ntype_subset = None

    # Pred ntypes
    if isinstance(hparams.pred_ntypes, str):
        assert len(hparams.pred_ntypes)
        pred_ntypes = hparams.pred_ntypes.split(" ")
    elif isinstance(hparams.pred_ntypes, Iterable):
        pred_ntypes = hparams.pred_ntypes
    else:
        raise Exception("Must provide `hparams.pred_ntypes` as a space-delimited string")

    # Exclude etype
    exclude_etypes = [(head_ntype, 'associated', go_ntype) for go_ntype in pred_ntypes]
    if 'exclude_etypes' in hparams and hparams.exclude_etypes:
        exclude_etypes_ = [etype.split(".") if isinstance(etype, str) else etype \
                           for etype in (hparams.exclude_etypes.split(" ") \
                                             if isinstance(hparams.exclude_etypes, str) \
                                             else hparams.exclude_etypes)]
        exclude_etypes.extend(exclude_etypes_)

    # Set GO etypes to include
    if 'go_etypes' in hparams and isinstance(hparams.go_etypes, str):
        go_etypes = hparams.go_etypes.split(" ") if len(hparams.go_etypes) else []
    elif 'go_etypes' in hparams and isinstance(hparams.go_etypes, Iterable):
        go_etypes = hparams.go_etypes
    else:
        go_etypes = None

    # Determine whether to include or exclude go_etypes
    if ntype_subset:
        pred_ntypes_in_graph = {'biological_process', 'cellular_component', 'molecular_function'}.intersection(
            ntype_subset)
        if not pred_ntypes_in_graph and go_etypes:
            hparams.go_etypes = go_etypes = None
        elif pred_ntypes_in_graph and not go_etypes:
            hparams.go_etypes = go_etypes = ['is_a', 'part_of', 'has_part']  # TODO add 'regulates'

    return add_parents, deepgraphgo_path, exclude_etypes, feature, go_etypes, head_ntype, labels_dataset, ntype_subset, \
           pred_ntypes, uniprotgoa_path, use_reverse


def get_DeepGraphGO_split(annot_df: pd.DataFrame, deepgraphgo_data: str, target='go_id',
                          pred_ntypes=['cc', 'bp', 'mf']):
    # Add GOA's from DeepGraphGO to UniProtGOA
    annot_df[target] = annot_df[target].map(to_list_of_strs)

    dgg_go_id = load_protein_dataset(deepgraphgo_data, namespaces=pred_ntypes)
    # Set train/valid/test_mask
    mask_cols = ['train_mask', 'valid_mask', 'test_mask']
    if not annot_df.columns.intersection(mask_cols).size:
        annot_df = annot_df.join(dgg_go_id[mask_cols], on='protein_id')
    annot_df[mask_cols] = annot_df[mask_cols].fillna(False)
    unmarked_nodes = ~annot_df[mask_cols].any(axis=1)
    annot_df.loc[unmarked_nodes, mask_cols] = \
        annot_df.loc[unmarked_nodes, mask_cols].replace({'train_mask': {False: True}})

    dgg_go_id[target] = dgg_go_id[target].map(to_list_of_strs)
    annot_df[target] = annot_df[target].apply(lambda d: d if isinstance(d, list) else []) + dgg_go_id[target]
    annot_df[target] = annot_df[target].map(np.unique).map(list)

    # Set train/valid/test split based on DeepGraphGO
    train_nodes = set(annot_df.query('train_mask == True').index)
    valid_nodes = set(annot_df.query('valid_mask == True').index)
    test_nodes = set(annot_df.query('test_mask == True').index)

    return annot_df, train_nodes, valid_nodes, test_nodes


def _replace_alt_id(li: List, mapper: Mapping[str, str]) -> List[str]:
    assert isinstance(mapper, dict), f"{type(mapper)}"
    if isinstance(li, str):
        return mapper.get(li, li)
    elif isinstance(li, Iterable):
        return [mapper.get(x, x) for x in li]
    else:
        return li
