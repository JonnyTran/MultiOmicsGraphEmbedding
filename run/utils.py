import logging
import os

import dill
import torch
from cogdl.datasets.gtn_data import ACM_GTNDataset, DBLP_GTNDataset, IMDB_GTNDataset
from cogdl.datasets.han_data import ACM_HANDataset, DBLP_HANDataset, IMDB_HANDataset
from ogb.linkproppred import PygLinkPropPredDataset, DglLinkPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset, DglNodePropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset, DglGraphPropPredDataset
from torch_geometric.datasets import AMiner

import moge
import moge.data.PyG.triplet_generator
from moge.data.dgl.graph_generator import DGLGraphSampler
from moge.data import HeteroNeighborGenerator
from moge.module.utils import preprocess_input


def load_node_dataset(dataset, method, hparams, train_ratio=None, dir_path="~/Bioinformatics_ExternalData/OGB/"):
    if "ogbn" in dataset:
        ogbn = PygNodePropPredDataset(name=dataset, root=dir_path)
        dataset = HeteroNeighborGenerator(ogbn, neighbor_sizes=hparams.neighbor_sizes,
                                          edge_dir="in",
                                          add_reverse_metapaths=hparams.use_reverse, inductive=hparams.inductive)
        if os.path.exists(ogbn.processed_dir + "/features.pk"):
            features = dill.load(open(ogbn.processed_dir + "/features.pk", 'rb'))

            # if (hparams.trainable_embedding if "trainable_embedding" in hparams else False):
            #     node_emb_init = {}
            #     for ntype, ndata in preprocess_input(features, device="cpu", dtype=torch.float).items():
            #         if ntype not in dataset.x_dict:
            #             node_emb_init[ntype] = ndata
            #         else:
            #             dataset.x_dict[ntype] = ndata
            #     hparams.embedding_dim = ndata.size(1)
            #     hparams.node_emb_init = node_emb_init
            #     print("added hparams.node_emb_init")
            # else:
            dataset.x_dict = preprocess_input(features, device="cpu", dtype=torch.float)
            print('added features')

        else:
            print("features.pk not found")

    elif dataset == "ACM":
        if method == "HAN" or method == "MetaPath2Vec":
            dataset = HeteroNeighborGenerator(ACM_HANDataset(), [25, 20], node_types=["P"],
                                              metapaths=["PAP", "PSP"] if "LATTE" in method else None,
                                              add_reverse_metapaths=True,
                                              head_node_type="P", resample_train=train_ratio,
                                              inductive=hparams.inductive)
        else:
            dataset = HeteroNeighborGenerator(ACM_GTNDataset(), [25, 20], node_types=["P"],
                                              metapaths=["PAP", "PA_P", "PSP", "PS_P"] if "LATTE" in method else None,
                                              add_reverse_metapaths=False,
                                              head_node_type="P", resample_train=train_ratio,
                                              inductive=hparams.inductive)

    elif dataset == "DBLP":
        if method == "HAN":
            dataset = HeteroNeighborGenerator(DBLP_HANDataset(), [25, 20],
                                              node_types=["A"], head_node_type="A", metapaths=None,
                                              add_reverse_metapaths=True,
                                              resample_train=train_ratio, inductive=hparams.inductive)
        elif "LATTE" in method:
            dataset = HeteroNeighborGenerator(DBLP_HANDataset(), [25, 20],
                                              node_types=["A", "P", "C", "T"], head_node_type="A",
                                              metapaths=["AC", "AP", "AT"],
                                              add_reverse_metapaths=True,
                                              resample_train=train_ratio, inductive=hparams.inductive)
        else:
            dataset = HeteroNeighborGenerator(DBLP_GTNDataset(), [25, 20], node_types=["A"], head_node_type="A",
                                              metapaths=["APA", "AP_A", "ACA", "AC_A"] if "LATTE" in method else None,
                                              add_reverse_metapaths=False,
                                              resample_train=train_ratio, inductive=hparams.inductive)

    elif dataset == "IMDB":
        if method == "HAN" or method == "MetaPath2Vec":
            dataset = HeteroNeighborGenerator(IMDB_HANDataset(), [25, 20], node_types=["M"],
                                              metapaths=["MAM", "MDM", "MWM"] if "LATTE" in method else None,
                                              add_reverse_metapaths=True,
                                              head_node_type="M",
                                              resample_train=train_ratio, inductive=hparams.inductive)
        else:
            dataset = HeteroNeighborGenerator(IMDB_GTNDataset(), neighbor_sizes=[25, 20], node_types=["M"],
                                              metapaths=["MDM", "MD_M", "MAM", "MA_M"] if "LATTE" in method else None,
                                              add_reverse_metapaths=False,
                                              head_node_type="M", inductive=hparams.inductive)
    elif dataset == "AMiner":
        dataset = HeteroNeighborGenerator(AMiner("datasets/aminer"), [25, 20], node_types=None,
                                          metapaths=[('paper', 'written by', 'author'),
                                                     ('venue', 'published', 'paper')], head_node_type="author",
                                          resample_train=train_ratio, inductive=hparams.inductive)
    elif dataset == "BlogCatalog":
        dataset = HeteroNeighborGenerator("datasets/blogcatalog6k.mat", [25, 20], node_types=["user", "tag"],
                                          head_node_type="user", resample_train=train_ratio,
                                          inductive=hparams.inductive)
    else:
        raise Exception(f"dataset {dataset} not found")
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
