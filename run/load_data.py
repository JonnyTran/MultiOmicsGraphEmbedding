import logging
import os
import pickle
from argparse import Namespace
from typing import Union

import dgl
import numpy as np
import pandas as pd
from cogdl.datasets.gtn_data import GTNDataset
from ogb.graphproppred import DglGraphPropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset
from openomics.database.ontology import GeneOntology, UniProtGOA
from torch_geometric.datasets import AMiner

import moge
import moge.dataset.PyG.triplet_generator
from moge.dataset.PyG.hetero_generator import HeteroLinkPredDataset, HeteroNodeClfDataset
from moge.dataset.dgl.graph_generator import DGLGraphSampler
from moge.dataset.dgl.node_generator import HeteroNeighborGenerator, DGLNodeGenerator
from moge.dataset.sequences import SequenceTokenizers
from moge.dataset.utils import get_edge_index_dict
from moge.model.dgl.DeepGraphGO import load_protein_dataset
from moge.model.dgl.NARS.data import load_acm, load_mag
from run.utils import add_node_embeddings


def load_node_dataset(name: str, method, args: Namespace, train_ratio=None,
                      dataset_path="dataset"):
    if name == "NARS":
        use_reverse = False
    else:
        use_reverse = True

    head_node_type = args.head_node_type

    if "ogbn" in name and method == "NARS":
        dataset = DGLNodeGenerator.from_heteronetwork(*load_mag(args=args),
                                                      inductive=args.inductive,
                                                      neighbor_sizes=args.neighbor_sizes, head_node_type="paper",
                                                      add_reverse_metapaths=False,
                                                      reshuffle_train=train_ratio if train_ratio else False)

    elif "ogbn" in name:

        ogbn = DglNodePropPredDataset(name=name, root=dataset_path)
        dataset = DGLNodeGenerator(ogbn,
                                   sampler="MultiLayerNeighborSampler",
                                   neighbor_sizes=args.neighbor_sizes,
                                   edge_dir="in",
                                   add_reverse_metapaths=use_reverse,
                                   inductive=args.inductive,
                                   reshuffle_train=train_ratio if train_ratio else False)

        if name == "ogbn-mag":
            add_node_embeddings(dataset, path=os.path.join(args.use_emb, "TransE_mag/"), args=args)
        elif name == "ogbn-proteins":
            add_node_embeddings(dataset, path=os.path.join(args.use_emb, "TransE_l2_ogbn-proteins/"), args=args)
        else:
            print(f"Cannot load embeddings for {dataset} at {args.use_emb}")

    elif name == "GTeX":
        with open(os.path.join(dataset_path, 'gtex_rna_ppi_multiplex_network.pickle'), "rb") as file:
            network = pickle.load(file)

        min_count = 50
        label_col = 'go_id'
        dataset = DGLNodeGenerator.from_heteronetwork(
            *network.to_dgl_heterograph(target=label_col, min_count=min_count, sequence=False),
            sampler="MultiLayerNeighborSampler",
            neighbor_sizes=args.neighbor_sizes,
            head_node_type=head_node_type,  # network.node_types if "LATTE" in method else "MessengerRNA",
            edge_dir="in",
            add_reverse_metapaths=use_reverse,
            inductive=False,
            classes=network.feature_transformer[label_col].classes_)
        dataset._name = f"GTeX-{label_col}@{dataset.n_classes}"

        add_node_embeddings(dataset, path=os.path.join(args.use_emb, "TransE_l2_GTeX/"), args=args)

        # Set up graph of the clasification labels
        if "LATTE" in method and "cls_graph" in method:
            print("adding cls_graph")
            geneontology = GeneOntology()

            all_go = set(geneontology.network.nodes)
            next_go = set(all_go) - set(dataset.classes)
            nodes = np.concatenate([dataset.classes, np.array(list(next_go))])

            edge_types = {e for u, v, e in geneontology.network.edges}
            edge_index_dict = geneontology.to_scipy_adjacency(nodes=nodes, edge_types=edge_types)

            args.classes = dataset.classes
            args.cls_graph = dgl.heterograph(edge_index_dict)

    elif name == "ACM":
        dataset = DGLNodeGenerator.from_heteronetwork(
            *load_acm(use_emb=os.path.join(args.use_emb, "TransE_acm/")),
            sampler="MultiLayerNeighborSampler",
            neighbor_sizes=args.neighbor_sizes,
            head_node_type="paper",
            edge_dir="in",
            add_reverse_metapaths=use_reverse,
            inductive=args.inductive, reshuffle_train=train_ratio if train_ratio else False)
        dataset._name = "ACM"

    elif name == "DBLP":
        dataset = DGLNodeGenerator.from_cogdl_graph(GTNDataset(root="./data/", name="gtn-dblp"),
                                                    neighbor_sizes=args.neighbor_sizes,
                                                    sampler="MultiLayerNeighborSampler",
                                                    node_types=["P", "A", "C"],
                                                    head_node_type="A",
                                                    metapaths=[("P", "PA", "A"), ("A", "AP", "P"), ("P", "PC", "C"),
                                                               ("C", "CP", "P")],
                                                    add_reverse_metapaths=False, inductive=args.inductive,
                                                    reshuffle_train=train_ratio if train_ratio else False)
        dataset._name = "DBLP"

    elif name == "IMDB":
        dataset = DGLNodeGenerator.from_cogdl_graph(GTNDataset(root="./data/", name="gtn-imdb"),
                                                    neighbor_sizes=args.neighbor_sizes,
                                                    sampler="MultiLayerNeighborSampler",
                                                    node_types=["M", "D", "A"],
                                                    head_node_type="M",
                                                    metapaths=[("D", "DM", "M"),
                                                               ("M", "AM", "D"),
                                                               ("D", "DA", "A"),
                                                               ("A", "AD", "D")
                                                               ],
                                                    add_reverse_metapaths=False, inductive=args.inductive,
                                                    reshuffle_train=train_ratio if train_ratio else False)
        dataset._name = "IMDB"

    elif name == "AMiner":
        dataset = HeteroNeighborGenerator(AMiner("datasets/aminer"), args.neighbor_sizes, node_types=None,
                                          metapaths=[('paper', 'written by', 'author'),
                                                     ('venue', 'published', 'paper')], head_node_type="author",
                                          resample_train=train_ratio, inductive=args.inductive)
    elif name == "BlogCatalog":
        dataset = HeteroNeighborGenerator("datasets/blogcatalog6k.mat", args.neighbor_sizes, node_types=["user", "tag"],
                                          head_node_type="user", resample_train=train_ratio,
                                          inductive=args.inductive)

    elif name == 'DeepGraphGO+UniProtGOA' and ".pickle" in dataset_path:
        with open(dataset_path, "rb") as file:
            network = pickle.load(file)
        geneontology = UniProtGOA()

        annot_df = network.annotations[head_node_type]
        annot_df['go_id'] = annot_df['go_id'].map(lambda d: d if isinstance(d, (list, np.ndarray)) else [])

        # Add GO interactions
        all_go = set(geneontology.network.nodes).intersection(geneontology.data.index)
        go_nodes = np.array(list(all_go))
        network.add_edges_from_ontology(geneontology, nodes=go_nodes, etypes=["is_a", "part_of"])

        uniprot_go_id = load_protein_dataset("../DeepGraphGO/data", namespaces=['mf', 'bp', 'cc'])
        uniprot_go_id['go_id'] = uniprot_go_id['go_id'].map(lambda d: d if isinstance(d, (list, np.ndarray)) else [])

        annot_df = annot_df.join(uniprot_go_id[['train_mask', 'valid_mask', 'test_mask']], on='protein_id')

        annot_df['go_id'] = annot_df['go_id'] + uniprot_go_id.loc[annot_df.index, 'go_id']
        annot_df['go_id'] = annot_df['go_id'].map(np.unique).map(list)

        network.train_nodes[head_node_type] = set(annot_df.query('train_mask == True').index)
        network.valid_nodes[head_node_type] = set(annot_df.query('valid_mask == True').index)
        network.test_nodes[head_node_type] = set(annot_df.query('test_mask == True').index)

        network.set_edge_traintest_mask(network.train_nodes, network.valid_nodes, network.test_nodes)

        if hasattr(args, 'sequence') and args.sequence:
            sequence_tokenizers = SequenceTokenizers(
                vocabularies={"MicroRNA": "armheb/DNA_bert_3",
                              "LncRNA": "armheb/DNA_bert_6",
                              "MessengerRNA": "armheb/DNA_bert_6",
                              'Protein': 'zjukg/OntoProtein',
                              'GO_term': "dmis-lab/biobert-base-cased-v1.2", },
                max_length=args.max_length)
            use_sequence = True
        else:
            sequence_tokenizers = None
            use_sequence = False

        namespaces = set(args.namespaces) if not isinstance(args.namespaces, (list, set, tuple)) else args.namespaces
        go_classes = geneontology.data.index[geneontology.data['namespace'].str.get(0).isin(namespaces)]

        if args.neighbor_loader == "HGTLoader":
            neighbor_sizes = [args.n_neighbors // 8] * args.t_order
        else:
            neighbor_sizes = [args.n_neighbors] * args.t_order

        dataset = HeteroNodeClfDataset.from_heteronetwork(
            network, target="go_id",
            labels_subset=geneontology.data.index.intersection(go_classes),
            min_count=50,
            expression=True,
            sequence=use_sequence, add_reverse_metapaths=use_reverse,
            head_node_type=args.head_node_type,
            neighbor_loader=args.neighbor_loader,
            neighbor_sizes=neighbor_sizes,
            split_namespace=True, go_ntype="GO_term",
            seq_tokenizer=sequence_tokenizers,
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

        if hasattr(args, 'cls_graph') and args.cls_graph:
            cls_network = dgl.heterograph(
                get_edge_index_dict(geneontology.network, nodes=dataset.classes, format="dgl"))
            args.cls_graph = cls_network

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

        dataset.add_edges_from_ontology(geneontology, etypes=["is_a", "part_of"], train_date=train_date,
                                        valid_date=valid_date, test_date="2021-12-31")
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
