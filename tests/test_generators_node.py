import logging
import pytest, random

import torch

from ogb.linkproppred import PygLinkPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
from cogdl.datasets.han_data import ACM_HANDataset, DBLP_HANDataset, IMDB_HANDataset

import moge
from moge.generator import HeteroNeighborSampler
from moge.module.utils import tensor_sizes
from moge.generator.utils import edge_dict_intersection, edge_sizes

dataset_path = "/home/jonny/Bioinformatics_ExternalData/OGB/"


@pytest.fixture
def get_dataset_ogb_hetero():
    ogbn = PygNodePropPredDataset(name="ogbn-mag", root=dataset_path)
    return ogbn


@pytest.fixture
def get_sampler_hetero(get_dataset_ogb_hetero):
    dataset = HeteroNeighborSampler(get_dataset_ogb_hetero, neighbor_sizes=[20, 10],
                                    head_node_type="paper",
                                    directed=True,
                                    add_reverse_metapaths=True)
    return dataset


@pytest.fixture
def get_dataset_ogb_homo():
    return DBLP_HANDataset()


@pytest.fixture
def get_sampler_homo(get_dataset_ogb_homo):
    dataset = HeteroNeighborSampler(DBLP_HANDataset(), neighbor_sizes=[25, 20],
                                    node_types=["A", "P", "C", "T"], head_node_type="A",
                                    metapaths=["AC", "AP", "AT"],
                                    add_reverse_metapaths=True, inductive=True)
    dataset.x_dict["P"] = dataset.x_dict["A"]
    dataset.x_dict["C"] = dataset.x_dict["A"]
    dataset.x_dict["T"] = dataset.x_dict["A"]
    return dataset


def test_generator_hetero(get_sampler_hetero):
    X, y, _ = get_sampler_hetero.sample(random.sample(get_sampler_hetero.training_idx.numpy().tolist(), 50),
                                        mode="train")
    print(tensor_sizes({"X": X, "y": y}))
    assert X is not None
    assert y is not None

    X, y, _ = get_sampler_hetero.sample(random.sample(get_sampler_hetero.validation_idx.numpy().tolist(), 50),
                                        mode="valid")
    print(tensor_sizes({"X": X, "y": y}))
    assert X is not None
    assert y is not None


def test_generator_homo(get_sampler_homo):
    X, y, _ = get_sampler_homo.sample(random.sample(get_sampler_homo.training_idx.numpy().tolist(), 50),
                                      mode="train")
    print(tensor_sizes({"X": X, "y": y}))
    assert X is not None
    assert y is not None

    X, y, _ = get_sampler_homo.sample(random.sample(get_sampler_homo.validation_idx.numpy().tolist(), 50),
                                      mode="valid")
    print(tensor_sizes({"X": X, "y": y}))
    assert X is not None
    assert y is not None


def test_sampled_edges_exists_hetero(get_sampler_hetero):
    node_idx = torch.randint(sum(get_sampler_hetero.num_nodes_dict.values()), (100,))
    batch_size, n_id, adjs = get_sampler_hetero.graph_sampler.sample(node_idx)

    global_node_index = get_sampler_hetero.get_local_node_index(adjs, n_id, )

    edge_index = get_sampler_hetero.get_local_edge_index_dict(adjs, n_id, global_node_index, filter_nodes=False)

    edge_index = {k: torch.stack([global_node_index[k[0]][v[0]], global_node_index[k[-1]][v[1]]], axis=0) \
                  for k, v in edge_index.items()}

    edge_counts = edge_sizes(edge_index)
    intersection_counts = edge_sizes(edge_dict_intersection(edge_index, get_sampler_hetero.edge_index_dict))

    assert edge_counts == intersection_counts


def test_sampled_edges_exists_homo(get_sampler_homo):
    node_idx = torch.randint(sum(get_sampler_homo.num_nodes_dict.values()), (100,))
    batch_size, n_id, adjs = get_sampler_homo.graph_sampler.sample(node_idx)

    global_node_index = get_sampler_homo.get_local_node_index(adjs, n_id, )

    edge_index = get_sampler_homo.get_local_edge_index_dict(adjs, n_id, global_node_index, filter_nodes=False)

    edge_index = {k: torch.stack([global_node_index[k[0]][v[0]], global_node_index[k[-1]][v[1]]], axis=0) \
                  for k, v in edge_index.items()}

    edge_counts = edge_sizes(edge_index)
    intersection_counts = edge_sizes(edge_dict_intersection(edge_index, get_sampler_homo.edge_index_dict))

    assert edge_counts == intersection_counts
