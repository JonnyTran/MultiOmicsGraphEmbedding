import os
import dill
from cogdl.datasets.gtn_data import ACM_GTNDataset, DBLP_GTNDataset, IMDB_GTNDataset
from cogdl.datasets.han_data import ACM_HANDataset, DBLP_HANDataset, IMDB_HANDataset
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset
import torch
from torch_geometric.datasets import AMiner

from moge.generator import HeteroNeighborSampler, TripletSampler, EdgeSampler
from moge.module.utils import preprocess_input

def load_node_dataset(name, method, train_ratio=None, hparams=None, dir_path="~/Bioinformatics_ExternalData/OGB/"):
    if "ogbn" in name:
        ogbn = PygNodePropPredDataset(name=name, root=dir_path)
        dataset = HeteroNeighborSampler(ogbn, neighbor_sizes=hparams.neighbor_sizes, directed=True, resample_train=None,
                                        add_reverse_metapaths=hparams.use_reverse, inductive=hparams.inductive)
        if os.path.exists(ogbn.processed_dir + "/features.pk"):
            features = dill.load(open(ogbn.processed_dir + "/features.pk", 'rb'))
            dataset.x_dict = preprocess_input(features, device="cpu", dtype=torch.float)
            print('added features')
        else:
            print("features.pk not found")

    elif name == "ACM":
        if method == "HAN" or method == "MetaPath2Vec":
            dataset = HeteroNeighborSampler(ACM_HANDataset(), [25, 10], node_types=["P"],
                                            metapaths=["PAP", "PSP"] if "LATTE" in method else None,
                                            add_reverse_metapaths=True,
                                            head_node_type="P", resample_train=train_ratio, inductive=hparams.inductive)
        else:
            dataset = HeteroNeighborSampler(ACM_GTNDataset(), [25, 10], node_types=["P"],
                                            metapaths=["PAP", "PA_P", "PSP", "PS_P"] if "LATTE" in method else None,
                                            add_reverse_metapaths=False,
                                            head_node_type="P", resample_train=train_ratio, inductive=hparams.inductive)

    elif name == "DBLP":
        if method == "HAN" or "LATTE" in method:
            dataset = HeteroNeighborSampler(DBLP_HANDataset(), [25, 10], node_types=["A"],
                                            metapaths=["ACA", "APA", "ATA"] if "LATTE" in method else None,
                                            head_node_type="A",
                                            add_reverse_metapaths=True,
                                            resample_train=train_ratio, inductive=hparams.inductive)
        else:
            dataset = HeteroNeighborSampler(DBLP_GTNDataset(), [25, 10], node_types=["A"],
                                            metapaths=["APA", "AP_A", "ACA", "AC_A"] if "LATTE" in method else None,
                                            head_node_type="A",
                                            add_reverse_metapaths=False,
                                            resample_train=train_ratio, inductive=hparams.inductive)

    elif name == "IMDB":
        if method == "HAN" or method == "MetaPath2Vec":
            dataset = HeteroNeighborSampler(IMDB_HANDataset(), [25, 10], node_types=["M"],
                                            metapaths=["MAM", "MDM", "MWM"] if "LATTE" in method else None,
                                            add_reverse_metapaths=True,
                                            head_node_type="M",
                                            resample_train=train_ratio, inductive=hparams.inductive)
        else:
            dataset = HeteroNeighborSampler(IMDB_GTNDataset(), neighbor_sizes=[25, 10], node_types=["M"],
                                            metapaths=["MDM", "MD_M", "MAM", "MA_M"] if "LATTE" in method else None,
                                            add_reverse_metapaths=False,
                                            head_node_type="M", inductive=hparams.inductive)
    elif name == "AMiner":
        dataset = HeteroNeighborSampler(AMiner("datasets/aminer"), [25, 10], node_types=None,
                                        metapaths=[('paper', 'written by', 'author'),
                                                   ('venue', 'published', 'paper')], head_node_type="author",
                                        resample_train=train_ratio, inductive=hparams.inductive)
    elif name == "BlogCatalog":
        dataset = HeteroNeighborSampler("datasets/blogcatalog6k.mat", [25, 10], node_types=["user", "tag"],
                                        head_node_type="user", resample_train=train_ratio, inductive=hparams.inductive)
    else:
        raise Exception(f"dataset {name} not found")
    return dataset


def load_link_dataset(name, hparams, path="~/Bioinformatics_ExternalData/OGB/"):
    if "ogbl" in name:
        ogbl = PygLinkPropPredDataset(name=name, root=path)

        if isinstance(ogbl, PygLinkPropPredDataset) and not hasattr(ogbl[0], "edge_index_dict") \
                and not hasattr(ogbl[0], "edge_reltype"):
            dataset = EdgeSampler(ogbl, directed=True, add_reverse_metapaths=hparams.use_reverse)
            print(dataset.node_types, dataset.metapaths)
        else:
            dataset = TripletSampler(ogbl, directed=True,
                                     head_node_type=None,
                                     add_reverse_metapaths=hparams.use_reverse)
            print(dataset.node_types, dataset.metapaths)
    else:
        raise Exception(f"dataset {name} not found")

    return dataset
