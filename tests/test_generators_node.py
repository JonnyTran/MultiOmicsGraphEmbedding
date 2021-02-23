import logging
import pytest

import torch

from ogb.linkproppred import PygLinkPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset

import moge
from moge.generator import HeteroNeighborSampler
from moge.module.utils import tensor_sizes
from moge.generator.utils import edge_dict_intersection

dataset_path = "/home/jonny/Bioinformatics_ExternalData/OGB/"


@pytest.fixture
def get_dataset_ogb_hetero():
    ogbn = PygNodePropPredDataset(name="ogbn-mag", root=dataset_path)
    return ogbn


@pytest.fixture
def get_sampler_hetero(get_dataset_ogb_hetero):
    dataset = HeteroNeighborSampler(get_dataset_ogb_hetero, neighbor_sizes=[20, 10],
                                    #                                 head_node_type="paper",
                                    directed=True,
                                    add_reverse_metapaths=True)
    return dataset


@pytest.fixture
def get_dataset_ogb_homo():
    ddi = PygLinkPropPredDataset(name="ogbl-collab", root="datasets")
    return ddi


@pytest.fixture
def get_sampler_homo(get_dataset_ogb_homo):
    dataset = moge.generator.PyG.edge_sampler.BidirectionalSampler(get_dataset_ogb_homo, neighbor_sizes=[10, 5],
                                                                   directed=True,
                                                                   add_reverse_metapaths=False)
    return dataset


def test_generator_hetero(get_sampler_hetero):
    X, y, z = get_sampler_hetero.sample(get_sampler_hetero.training_idx[:50], mode="train")
    print(tensor_sizes({"X": X, "y": y, "z": z}))
    assert len(X) + len(y) + len(z) >= 3

    X, y, z = get_sampler_hetero.sample(get_sampler_hetero.validation_idx[:50], mode="valid")
    print(tensor_sizes({"X": X, "y": y, "z": z}))
    assert len(X) + len(y) + len(z) >= 3


def test_generator_homo(get_sampler_homo):
    X, y, z = get_sampler_homo.sample(get_sampler_homo.training_idx[:50], mode="train")
    print(tensor_sizes({"X": X, "y": y, "z": z}))
    assert len(X) + len(y) + len(z) >= 3

    X, y, z = get_sampler_homo.sample(get_sampler_homo.validation_idx[:50],
                                      mode="valid")
    print(tensor_sizes({"X": X, "y": y, "z": z}))
    assert len(X) + len(y) + len(z) >= 3


def test_sampled_edges_exists_hetero(get_sampler_hetero):
    node_idx = torch.randint(len(get_sampler_hetero.node_type), (100,))
    batch_size, n_id, adjs = get_sampler_hetero.neighbor_sampler.sample(node_idx)

    global_node_index = get_sampler_hetero.get_local_node_index(adjs, n_id, )

    edge_index = get_sampler_hetero.get_local_edge_index_dict(adjs, n_id, global_node_index, filter_nodes=False)

    edge_index = {k: torch.stack([global_node_index[k[0]][v[0]], global_node_index[k[-1]][v[1]]], axis=0) \
                  for k, v in edge_index.items()}

    edge_counts = {k: v.shape[1] for k, v in edge_index.items()}
    intersection_counts = edge_dict_intersection(edge_index, get_sampler_hetero.edge_index_dict)

    assert edge_counts == intersection_counts


def test_sampled_edges_exists_homo(get_sampler_homo):
    node_idx = torch.randint(len(get_sampler_homo.node_type), (100,))
    batch_size, n_id, adjs = get_sampler_homo.neighbor_sampler.sample(node_idx)

    global_node_index = get_sampler_homo.get_local_node_index(adjs, n_id, )

    edge_index = get_sampler_homo.get_local_edge_index_dict(adjs, n_id, global_node_index, filter_nodes=False)

    edge_index = {k: torch.stack([global_node_index[k[0]][v[0]], global_node_index[k[-1]][v[1]]], axis=0) \
                  for k, v in edge_index.items()}

    edge_counts = {k: v.shape[1] for k, v in edge_index.items()}
    intersection_counts = edge_dict_intersection(edge_index, get_sampler_homo.edge_index_dict)

    assert edge_counts == intersection_counts
