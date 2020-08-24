from cogdl.datasets.gtn_data import ACM_GTNDataset, DBLP_GTNDataset, IMDB_GTNDataset
from cogdl.datasets.han_data import ACM_HANDataset, DBLP_HANDataset, IMDB_HANDataset
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.datasets import AMiner

from moge.generator import HeteroNeighborSampler, TripletSampler, EdgeSampler


def load_node_dataset(name, method, train_ratio=None, hparams=None, dir_path="~/Bioinformatics_ExternalData/OGB/"):
    if "ogbn" in name:
        ogbn = PygNodePropPredDataset(name=name, root=dir_path)
        dataset = HeteroNeighborSampler(ogbn, directed=True, neighbor_sizes=hparams.neighbor_sizes,
                                        node_types=list(ogbn[0].num_nodes_dict.keys()),
                                        head_node_type=None,
                                        add_reverse_metapaths=hparams.use_reverse, resample_train=None)
    elif name == "ACM":
        if method == "HAN" or method == "MetaPath2Vec":
            dataset = HeteroNeighborSampler(ACM_HANDataset(), node_types=["P"], metapaths=["PAP", "PLP"],
                                            head_node_type="P",
                                            resample_train=train_ratio)
        else:
            dataset = HeteroNeighborSampler(ACM_GTNDataset(), node_types=["P"], metapaths=["PAP", "PLP"],
                                            head_node_type="P",
                                            resample_train=train_ratio)

    elif name == "DBLP":
        if method == "HAN" or method == "MetaPath2Vec" or method == "LATTE":
            dataset = HeteroNeighborSampler(DBLP_HANDataset(), node_types=["A"], metapaths=["APA", "ACA", "ATA"],
                                            head_node_type="A",
                                            resample_train=train_ratio)
        else:
            dataset = HeteroNeighborSampler(DBLP_GTNDataset(), node_types=["A"], metapaths=["APA", "ACA", "ATA", "AGA"],
                                            head_node_type="A",
                                            resample_train=train_ratio)

    elif name == "IMDB":
        if method == "HAN" or method == "MetaPath2Vec":
            dataset = HeteroNeighborSampler(IMDB_HANDataset(), node_types=["M"], metapaths=["MAM", "MDM", "MYM"],
                                            head_node_type="M",
                                            resample_train=train_ratio)
        else:
            dataset = HeteroNeighborSampler(IMDB_GTNDataset(), node_types=["M"], metapaths=["MAM", "MDM", "MYM"],
                                            head_node_type="M",
                                            resample_train=train_ratio)
    elif name == "AMiner":
        dataset = HeteroNeighborSampler(AMiner("datasets/aminer"), node_types=None,
                                        metapaths=[('paper', 'written by', 'author'),
                                                   ('venue', 'published', 'paper')],
                                        head_node_type="author",
                                        resample_train=train_ratio)
    elif name == "BlogCatalog":
        dataset = HeteroNeighborSampler("datasets/blogcatalog6k.mat", node_types=["user", "tag"], head_node_type="user",
                                        resample_train=train_ratio)
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
                                     node_types=list(ogbl[0].num_nodes_dict.keys()) if hasattr(ogbl[0],
                                                                                               "num_nodes_dict") else None,
                                     head_node_type=None,
                                     add_reverse_metapaths=hparams.use_reverse)
            print(dataset.node_types, dataset.metapaths)
    else:
        raise Exception(f"dataset {name} not found")

    return dataset
