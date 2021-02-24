import logging
import pytest

import torch

from ogb.linkproppred import PygLinkPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset

import moge
from moge.generator import BidirectionalSampler
from moge.module.utils import tensor_sizes
from moge.generator.utils import edge_dict_intersection, edge_sizes

cohort_folder_path = "datasets"


@pytest.fixture
def get_dataset_ogb_hetero():
    biokg = PygLinkPropPredDataset(name="ogbl-biokg", root=cohort_folder_path)
    return biokg


@pytest.fixture
def get_sampler_homo(get_dataset_ogb_hetero):
    dataset = BidirectionalSampler(get_dataset_ogb_hetero, neighbor_sizes=[10, 5],
                                   directed=True,
                                   node_types=['protein', 'drug', 'function', 'disease', 'sideeffect'],
                                   head_node_type="protein",
                                   add_reverse_metapaths=False)
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


def test_generator_hetero(get_dataset_ogb_hetero):
    X, y, z = get_dataset_ogb_hetero.sample(get_dataset_ogb_hetero.training_idx[:50], mode="train")
    print(tensor_sizes({"X": X, "y": y, "z": z}))
    assert len(X) + len(y) + len(z) >= 3

    X, y, z = get_dataset_ogb_hetero.sample(get_dataset_ogb_hetero.validation_idx[:50], mode="valid")
    print(tensor_sizes({"X": X, "y": y, "z": z}))
    assert len(X) + len(y) + len(z) >= 3


def test_generator_homo(get_sampler_homo):
    X, y, z = get_sampler_homo.sample(get_sampler_homo.training_idx[:50], mode="train")
    print(tensor_sizes({"X": X, "y": y, "z": z}))
    assert len(X) + len(y) + len(z) >= 3

    X, y, z = get_sampler_homo.sample(get_sampler_homo.validation_idx[:50], mode="valid")
    print(tensor_sizes({"X": X, "y": y, "z": z}))
    assert len(X) + len(y) + len(z) >= 3


def test_sampled_edges_exists_hetero(get_dataset_ogb_hetero):
    node_idx = torch.randint(sum(get_dataset_ogb_hetero.num_nodes_dict.values()), (100,))
    batch_size, n_id, adjs = get_dataset_ogb_hetero.graph_sampler.sample(node_idx)
    global_node_index = get_dataset_ogb_hetero.graph_sampler.get_local_node_index(adjs, n_id, )
    edge_index = get_dataset_ogb_hetero.graph_sampler.get_local_edge_index_dict(adjs, n_id, global_node_index,
                                                                                filter_nodes=False)

    edge_index = {k: torch.stack([global_node_index[k[0]][v[0]], global_node_index[k[-1]][v[1]]], axis=0) \
                  for k, v in edge_index.items()}

    edge_counts = edge_sizes(edge_index)
    intersection_counts = edge_sizes(edge_dict_intersection(edge_index, get_dataset_ogb_hetero.edge_index_dict))

    assert edge_counts == intersection_counts


def test_sampled_edges_exists_homo(get_sampler_homo):
    node_idx = torch.randint(sum(get_sampler_homo.num_nodes_dict.values()), (100,))
    batch_size, n_id, adjs = get_sampler_homo.graph_sampler.sample(node_idx)
    global_node_index = get_sampler_homo.graph_sampler.get_local_node_index(adjs, n_id, )
    edge_index = get_sampler_homo.graph_sampler.get_local_edge_index_dict(adjs, n_id, global_node_index,
                                                                          filter_nodes=False)

    edge_index = {k: torch.stack([global_node_index[k[0]][v[0]], global_node_index[k[-1]][v[1]]], axis=0) \
                  for k, v in edge_index.items()}

    edge_counts = edge_sizes(edge_index)
    intersection_counts = edge_sizes(edge_dict_intersection(edge_index, get_sampler_homo.edge_index_dict))

    assert edge_counts == intersection_counts
