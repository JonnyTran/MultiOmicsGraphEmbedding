import logging
import os
import pickle
from argparse import Namespace

import dgl
import dill
import torch
from cogdl.datasets.gtn_data import ACM_GTNDataset, DBLP_GTNDataset, IMDB_GTNDataset, GTNDataset
from cogdl.datasets.han_data import ACM_HANDataset, DBLP_HANDataset, IMDB_HANDataset
from ogb.linkproppred import PygLinkPropPredDataset, DglLinkPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset, DglNodePropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset, DglGraphPropPredDataset
from torch_geometric.datasets import AMiner
from typing import Union

import moge
import moge.data.PyG.triplet_generator
from moge.data.dgl.graph_generator import DGLGraphSampler
from moge.data import HeteroNeighborGenerator, DGLNodeSampler
from moge.module.dgl.NARS.data import load_acm
from moge.module.utils import preprocess_input


def add_node_embeddings(dataset: Union[HeteroNeighborGenerator, DGLNodeSampler], path: str, skip_ntype: str = None,
                        args: Namespace = None):
    node_emb = {}
    if os.path.exists(path) and os.path.isdir(path):
        for file in os.listdir(path):
            ntype = file.split(".")[0]
            ndata = torch.load(os.path.join(path, file)).float()

            node_emb[ntype] = ndata

    elif os.path.exists(path) and os.path.isfile(path):
        features = dill.load(open(path, 'rb'))  # Assumes .pk file

        for ntype, ndata in preprocess_input(features, device="cpu", dtype=torch.float).items():
            node_emb[ntype] = ndata
    else:
        print(f"Failed to import embeddings from {path}")

    for ntype, ndata in node_emb.items():
        if skip_ntype == ntype:
            logging.info(f"Use original features (not embeddings) for node type: {ntype}")
            continue

        if "freeze_embeddings" in args and args.freeze_embeddings == False:
            print("got here")
            if "node_emb_init" not in args:
                args.node_emb_init = {}

            args.node_emb_init[ntype] = ndata

        elif isinstance(dataset.G, dgl.DGLHeteroGraph):
            dataset.G.nodes[ntype].data["feat"] = ndata

        elif isinstance(dataset.G, HeteroNeighborGenerator):
            dataset.G.x_dict[ntype] = ndata
        else:
            raise Exception(f"Cannot recognize type of {dataset.G}")

        print(f"Loaded embeddings for {ntype}: {ndata.shape}")


def load_node_dataset(name: str, method, args: Namespace, train_ratio=None,
                      dataset_path="dataset"):
    if "ogbn" in name:
        ogbn = DglNodePropPredDataset(name=name, root=dataset_path)
        dataset = DGLNodeSampler(ogbn,
                                 sampler="MultiLayerNeighborSampler",
                                 neighbor_sizes=args.neighbor_sizes,
                                 edge_dir="in",
                                 add_reverse_metapaths=True,
                                 inductive=args.inductive, reshuffle_train=train_ratio if train_ratio else False)

        if name == "ogbn-mag":
            add_node_embeddings(dataset, path=os.path.join(args.use_emb, "TransE_mag/"), args=args)
        elif name == "ogbn-proteins":
            add_node_embeddings(dataset, path=os.path.join(args.use_emb, "TransE_l2_ogbn-proteins/"), args=args)
        else:
            print(f"Cannot load embeddings for {dataset} at {args.use_emb}")

    elif name == "GTeX":
        with open(os.path.join(dataset_path, 'gtex_rna_ppi_multiplex_network.pickle'), "rb") as file:
            network = pickle.load(file)

        dataset = DGLNodeSampler.from_dgl_heterograph(*network.to_dgl_heterograph(min_count=500),
                                                      sampler="MultiLayerNeighborSampler",
                                                      neighbor_sizes=args.neighbor_sizes,
                                                      head_node_type=network.node_types,
                                                      edge_dir="in",
                                                      add_reverse_metapaths=True,
                                                      inductive=False)
        dataset._name = "GTeX"
        add_node_embeddings(dataset, path=os.path.join(args.use_emb, "TransE_l2_GTeX/"), args=args)

    elif name == "ACM":
        dataset = DGLNodeSampler.from_dgl_heterograph(
            *load_acm(use_emb=os.path.join(args.use_emb, "TransE_acm/")),
            sampler="MultiLayerNeighborSampler",
            neighbor_sizes=[10],
            head_node_type="paper",
            edge_dir="in",
            add_reverse_metapaths=True,
            inductive=args.inductive, reshuffle_train=train_ratio if train_ratio else False)
        dataset._name = "ACM"

    elif name == "DBLP":
        dataset = DGLNodeSampler.from_cogdl_graph(GTNDataset(root="./data/", name="gtn-dblp"),
                                                  neighbor_sizes=[25, 25],
                                                  sampler="MultiLayerNeighborSampler",
                                                  node_types=["P", "A", "C"],
                                                  head_node_type="A",
                                                  metapaths=[("P", "PA", "A"), ("A", "AP", "P"), ("P", "PC", "C"),
                                                             ("C", "CP", "P")],
                                                  add_reverse_metapaths=False, inductive=args.inductive,
                                                  reshuffle_train=train_ratio if train_ratio else False)
        dataset._name = "DBLP"

    elif name == "IMDB":
        dataset = DGLNodeSampler.from_cogdl_graph(GTNDataset(root="./data/", name="gtn-imdb"),
                                                  neighbor_sizes=[25, 25],
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
        dataset = HeteroNeighborGenerator(AMiner("datasets/aminer"), [25, 20], node_types=None,
                                          metapaths=[('paper', 'written by', 'author'),
                                                     ('venue', 'published', 'paper')], head_node_type="author",
                                          resample_train=train_ratio, inductive=args.inductive)
    elif name == "BlogCatalog":
        dataset = HeteroNeighborGenerator("datasets/blogcatalog6k.mat", [25, 20], node_types=["user", "tag"],
                                          head_node_type="user", resample_train=train_ratio,
                                          inductive=args.inductive)
    else:
        raise Exception(f"dataset {name} not found")

    return dataset


def load_link_dataset(name, hparams, path="~/Bioinformatics_ExternalData/OGB/"):
    if "ogbl" in name:
        ogbl = PygLinkPropPredDataset(name=name, root=path)

        if isinstance(ogbl, PygLinkPropPredDataset) and not hasattr(ogbl[0], "edge_index_dict") \
                and not hasattr(ogbl[0], "edge_reltype"):

            dataset = moge.generator.PyG.triplet_generator.BidirectionalGenerator(ogbl,
                                                                                  neighbor_sizes=[20, 15],
                                                                                  edge_dir=False,
                                                                                  add_reverse_metapaths=hparams.use_reverse)

        else:
            from moge.data import BidirectionalGenerator

            dataset = moge.generator.PyG.triplet_generator.BidirectionalGenerator(ogbl, edge_dir=True,
                                                                                  neighbor_sizes=[
                                                                                      hparams.n_neighbors_1],
                                                                                  negative_sampling_size=hparams.neg_sampling_ratio,
                                                                                  test_negative_sampling_size=500,
                                                                                  head_node_type=None,
                                                                                  add_reverse_metapaths=hparams.use_reverse)

        logging.info(
            f"ntypes: {dataset.node_types}, head_nt: {dataset.head_node_type}, metapaths: {dataset.metapaths}")

    else:
        raise Exception(f"dataset {name} not found")


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
