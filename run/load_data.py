import glob
import logging
import os
import pickle
from argparse import Namespace
from pathlib import Path
from typing import Union

import pandas as pd
from ogb.graphproppred import DglGraphPropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset
from ruamel import yaml
from torch_geometric.datasets import AMiner

import moge
import moge.dataset.PyG.triplet_generator
from moge.dataset.PyG.hetero_generator import HeteroLinkPredDataset
from moge.dataset.dgl.graph_generator import DGLGraphSampler
from moge.dataset.dgl.node_generator import HeteroNeighborGenerator, DGLNodeGenerator
from moge.dataset.sequences import SequenceTokenizers
from moge.model.dgl.NARS.data import load_acm, load_mag
from moge.network.hetero import HeteroNetwork
from openomics.database.ontology import GeneOntology
from run.datasets.uniprotgoa import build_uniprot_dataset
from run.utils import add_node_embeddings


def load_node_dataset(name: str, method, hparams: Namespace, train_ratio=None,
                      dataset_path: Path = "dataset"):
    if name == "NARS":
        use_reverse = False
    else:
        use_reverse = True

    if "ogbn" in name and method == "NARS":
        dataset = DGLNodeGenerator.from_heteronetwork(*load_mag(args=hparams),
                                                      inductive=hparams.inductive,
                                                      neighbor_sizes=hparams.neighbor_sizes, head_node_type="paper",
                                                      add_reverse_metapaths=False,
                                                      reshuffle_train=train_ratio if train_ratio else False)

    elif "ogbn" in name:

        ogbn = DglNodePropPredDataset(name=name, root=dataset_path)
        dataset = DGLNodeGenerator(ogbn,
                                   sampler="MultiLayerNeighborSampler",
                                   neighbor_sizes=hparams.neighbor_sizes,
                                   edge_dir="in",
                                   add_reverse_metapaths=use_reverse,
                                   inductive=hparams.inductive,
                                   reshuffle_train=train_ratio if train_ratio else False)

        if name == "ogbn-mag":
            add_node_embeddings(dataset, path=os.path.join(hparams.use_emb, "TransE_mag/"), args=hparams)
        elif name == "ogbn-proteins":
            add_node_embeddings(dataset, path=os.path.join(hparams.use_emb, "TransE_l2_ogbn-proteins/"), args=hparams)
        else:
            print(f"Cannot load embeddings for {dataset} at {hparams.use_emb}")

    elif name == "ACM":
        dataset = DGLNodeGenerator.from_heteronetwork(
            *load_acm(use_emb=os.path.join(hparams.use_emb, "TransE_acm/")),
            sampler="MultiLayerNeighborSampler",
            neighbor_sizes=hparams.neighbor_sizes,
            head_node_type="paper",
            edge_dir="in",
            add_reverse_metapaths=use_reverse,
            inductive=hparams.inductive, reshuffle_train=train_ratio if train_ratio else False)
        dataset._name = "ACM"
    elif name == "DBLP":
        from cogdl.datasets.gtn_data import GTNDataset
        dataset = DGLNodeGenerator.from_cogdl_graph(GTNDataset(root="./data/", name="gtn-dblp"),
                                                    neighbor_sizes=hparams.neighbor_sizes,
                                                    sampler="MultiLayerNeighborSampler",
                                                    node_types=["P", "A", "C"],
                                                    head_node_type="A",
                                                    metapaths=[("P", "PA", "A"), ("A", "AP", "P"), ("P", "PC", "C"),
                                                               ("C", "CP", "P")],
                                                    add_reverse_metapaths=False, inductive=hparams.inductive,
                                                    reshuffle_train=train_ratio if train_ratio else False)
        dataset._name = "DBLP"
    elif name == "IMDB":
        from cogdl.datasets.gtn_data import GTNDataset
        dataset = DGLNodeGenerator.from_cogdl_graph(GTNDataset(root="./data/", name="gtn-imdb"),
                                                    neighbor_sizes=hparams.neighbor_sizes,
                                                    sampler="MultiLayerNeighborSampler",
                                                    node_types=["M", "D", "A"],
                                                    head_node_type="M",
                                                    metapaths=[("D", "DM", "M"),
                                                               ("M", "AM", "D"),
                                                               ("D", "DA", "A"),
                                                               ("A", "AD", "D")
                                                               ],
                                                    add_reverse_metapaths=False, inductive=hparams.inductive,
                                                    reshuffle_train=train_ratio if train_ratio else False)
        dataset._name = "IMDB"
    elif name == "AMiner":
        dataset = HeteroNeighborGenerator(AMiner("datasets/aminer"), hparams.neighbor_sizes, node_types=None,
                                          metapaths=[('paper', 'written by', 'author'),
                                                     ('venue', 'published', 'paper')], head_node_type="author",
                                          resample_train=train_ratio, inductive=hparams.inductive)
    elif name == "BlogCatalog":
        dataset = HeteroNeighborGenerator("datasets/blogcatalog6k.mat", hparams.neighbor_sizes,
                                          node_types=["user", "tag"],
                                          head_node_type="user", resample_train=train_ratio,
                                          inductive=hparams.inductive)

    elif method == 'DeepGraphGO':
        dataset = None  # will load in method

    elif name in ["HUMAN_MOUSE", "MULTISPECIES"]:

        if name == 'HUMAN_MOUSE':
            dataset_path = '~/PycharmProjects/Multiplex-Graph-Embedding/data/heteronetwork/DGG_HUMAN_MOUSE_MirTarBase_TarBase_LncBase_RNAInter_STRINGsplit_BioGRID_mRNAprotein_transcriptlevel.network.pickle'
        elif name == "MULTISPECIES":
            dataset_path = '~/PycharmProjects/Multiplex-Graph-Embedding/data/heteronetwork/DGG_MirTarBase_TarBase_LncBase_RNAInter_STRINGsplit_BioGRID_mRNAprotein_transcriptlevel.network.pickle'

        mlb_path = os.path.expanduser('~/Bioinformatics_ExternalData/LATTE2GO')
        mlb_paths = glob.glob(f'{mlb_path}/{hparams.dataset}-{hparams.pred_ntypes}/go_id.mlb')
        if mlb_paths:
            hparams.mlb_path = mlb_paths[0]

        with open('run/configs/_latte2go_helper.yaml', 'r') as f:
            args = yaml.safe_load(f)
        hparams.__dict__.update(args)

        dataset = build_uniprot_dataset('UniProt', dataset_path=dataset_path, hparams=hparams)


    elif 'UNIPROT' in name.upper() and (
            isinstance(dataset_path, HeteroNetwork) or (isinstance(dataset_path, str) and ".pickle" in dataset_path)):
        dataset = build_uniprot_dataset(name, dataset_path=dataset_path, hparams=hparams)

    else:
        raise Exception(f"dataset {name} not found")

    return dataset


def load_link_dataset(name: str, hparams: Namespace, path="~/Bioinformatics_ExternalData/OGB/") -> \
        Union[PygLinkPropPredDataset, HeteroLinkPredDataset]:

    if "ogbl" in name:
        ogbl = PygLinkPropPredDataset(name=name, root=path)

        if isinstance(ogbl, PygLinkPropPredDataset) and not hasattr(ogbl[0], "edge_index_dict") \
                and not hasattr(ogbl[0], "edge_reltype"):

            dataset = moge.dataset.PyG.triplet_generator.BidirectionalGenerator(ogbl,
                                                                                neighbor_sizes=[20, 15],
                                                                                edge_dir=False,
                                                                                add_reverse_metapaths=hparams.use_reverse)

        else:

            dataset = moge.dataset.PyG.triplet_generator.BidirectionalGenerator(ogbl, edge_dir=True,
                                                                                neighbor_sizes=[
                                                                                    hparams.n_neighbors_1],
                                                                                negative_sampling_size=hparams.neg_sampling_ratio,
                                                                                test_negative_sampling_size=500,
                                                                                head_node_type=None,
                                                                                add_reverse_metapaths=hparams.use_reverse)

        logging.info(
            f"ntypes: {dataset.node_types}, head_nt: {dataset.head_node_type}, metapaths: {dataset.metapaths}")

    elif ".pickle" in path:
        with open(path, "rb") as file:
            network = pickle.load(file)

        if hasattr(hparams, 'sequence') and hparams.sequence:
            sequence_tokenizers = SequenceTokenizers(
                vocabularies={"MicroRNA": "armheb/DNA_bert_3",
                              "LncRNA": "armheb/DNA_bert_6",
                              "MessengerRNA": "armheb/DNA_bert_6",
                              'Protein': 'zjukg/OntoProtein',
                              'GO_term': "dmis-lab/biobert-base-cased-v1.2", },
                max_length=hparams.max_length)
            use_sequence = True
        else:
            sequence_tokenizers = None
            use_sequence = False

        hetero, classes, nodes = network.to_pyg_heterodata(target=None, min_count=None, sequence=use_sequence,
                                                           expression=False)

        n_neighbors = hparams.n_neighbors if hparams.neighbor_loader == "HGTLoader" else hparams.n_neighbors // 8
        dataset = HeteroLinkPredDataset.from_heteronetwork(hetero, classes, nodes,
                                                           negative_sampling_size=1000,
                                                           pred_metapaths=[],
                                                           head_node_type=hparams.head_node_type,
                                                           neighbor_loader=hparams.neighbor_loader,
                                                           neighbor_sizes=[n_neighbors] * hparams.t_order,
                                                           seq_tokenizer=sequence_tokenizers)

        train_date = hparams.train_date
        valid_date = pd.to_datetime(train_date) + pd.to_timedelta(52, "W")
        geneontology = GeneOntology(
            file_resources={"go-basic.obo": "http://purl.obolibrary.org/obo/go/go-basic.obo"} \
                if "mlm" in name else None)

        dataset.add_edges_from_ontology(geneontology, etypes=["is_a", "part_of"])
        dataset._name = "_".join([name, train_date])

        dataset.pred_metapaths = [(hparams.head_node_type, 'associated', 'biological_process'),
                                  (hparams.head_node_type, 'associated', 'cellular_component'),
                                  (hparams.head_node_type, 'associated', 'molecular_function')]
        dataset.ntype_mapping = {'biological_process': dataset.go_ntype,
                                 'cellular_component': dataset.go_ntype,
                                 'molecular_function': dataset.go_ntype}

    else:
        raise Exception(f"dataset {name} not found")

    hparams.n_classes = dataset.n_classes

    return dataset

def load_graph_dataset(name, hparams, path="~/Bioinformatics_ExternalData/OGB/"):
    if "ogbg" in name:
        ogbg = DglGraphPropPredDataset(name=name, root=path)

        dataset = DGLGraphSampler(ogbg,
                                  embedding_dim=hparams.embeddings_dim,
                                  add_self_loop=True if "add_self_loop" in hparams and hparams.add_self_loop else False,
                                  edge_dir="in",
                                  add_reverse_metapaths=hparams.use_reverse,
                                  inductive=hparams.inductive)

        logging.info(
            f"n_graphs: {len(ogbg.graphs)}, ntypes: {dataset.node_types}, head_nt: {dataset.head_node_type}, metapaths: {dataset.metapaths}")

    else:
        raise Exception(f"dataset {name} not found")

    return dataset
