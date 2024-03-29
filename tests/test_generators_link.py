import random

import pytest
import torch
from ogb.linkproppred import PygLinkPropPredDataset

import moge
import moge.dataset.PyG.edge_generator
import moge.dataset.PyG.triplet_generator
from moge.dataset.utils import edge_dict_intersection, edge_dict_sizes
from moge.dataset.utils import nonduplicate_indices
from moge.model.utils import tensor_sizes

cohort_folder_path = "datasets"


@pytest.fixture
def generate_dataset_hetero():
    biokg = PygLinkPropPredDataset(name="ogbl-biokg", root=cohort_folder_path)
    return biokg


@pytest.fixture
def generate_dataset_homo():
    ddi = PygLinkPropPredDataset(name="ogbl-collab", root="datasets")
    return ddi


@pytest.fixture
def generator_hetero(generate_dataset_hetero):
    dataset = moge.generator.PyG.triplet_generator.BidirectionalGenerator(generate_dataset_hetero,
                                                                          neighbor_sizes=[10, 5],
                                                                          edge_dir=True,
                                                                          node_types=['protein', 'drug', 'function',
                                                                                      'disease', 'sideeffect'],
                                                                          head_node_type="protein",
                                                                          add_reverse_metapaths=False)
    return dataset


@pytest.fixture
def generator_homo(generate_dataset_homo):
    dataset = moge.generator.PyG.edge_generator.BidirectionalGenerator(generate_dataset_homo, neighbor_sizes=[10, 5],
                                                                       edge_dir=True,
                                                                       add_reverse_metapaths=False)
    return dataset


def test_generator_hetero(generator_hetero):
    X, y, z = generator_hetero.sample(generator_hetero.training_idx[:50], mode="train")
    print(tensor_sizes({"X": X, "y": y, "z": z}))
    assert len(X) + len(y) + len(z) >= 3

    X, y, z = generator_hetero.sample(generator_hetero.validation_idx[:50], mode="valid")
    print(tensor_sizes({"X": X, "y": y, "z": z}))
    assert len(X) + len(y) + len(z) >= 3


def test_generator_homo(generator_homo):
    X, y, z = generator_homo.sample(generator_homo.training_idx[:50], mode="train")
    print(tensor_sizes({"X": X, "y": y, "z": z}))
    assert len(X) + len(y) + len(z) >= 3

    X, y, z = generator_homo.sample(generator_homo.validation_idx[:50], mode="valid")
    print(tensor_sizes({"X": X, "y": y, "z": z}))
    assert len(X) + len(y) + len(z) >= 3


def test_sampled_edges_exists_hetero(generator_hetero):
    node_idx = random.sample(range(generator_hetero.num_nodes_dict[generator_hetero.head_node_type]), 50)
    batch_size, n_id, adjs = generator_hetero.graph_sampler.sample(node_idx)
    global_node_index = generator_hetero.graph_sampler.get_local_nodes(n_id)
    edge_index = generator_hetero.graph_sampler.get_edge_index_dict(adjs, n_id, global_node_index,
                                                                    filter_nodes=False)

    edge_index = {k: torch.stack([global_node_index[k[0]][v[0]], global_node_index[k[-1]][v[1]]], dim=0) \
                  for k, v in edge_index.items()}

    edge_counts = edge_dict_sizes({k: eids[:, nonduplicate_indices(eids)] for k, eids in edge_index.items()})
    intersection_counts = edge_dict_sizes(edge_dict_intersection(edge_index, generator_hetero.edge_index_dict))

    assert edge_counts == intersection_counts


def test_sampled_edges_exists_homo(generator_homo):
    node_idx = random.sample(range(generator_homo.num_nodes_dict[generator_homo.head_node_type]), 50)
    batch_size, n_id, adjs = generator_homo.graph_sampler.sample(node_idx)
    global_node_index = generator_homo.graph_sampler.get_local_nodes(n_id)
    edge_index = generator_homo.graph_sampler.get_edge_index_dict(adjs, n_id, global_node_index,
                                                                  filter_nodes=False)

    edge_index = {k: torch.stack([global_node_index[k[0]][v[0]], global_node_index[k[-1]][v[1]]], dim=0) \
                  for k, v in edge_index.items()}

    edge_counts = edge_dict_sizes({k: eids[:, nonduplicate_indices(eids)] for k, eids in edge_index.items()})
    intersection_counts = edge_dict_sizes(edge_dict_intersection(edge_index, generator_homo.edge_index_dict))

    assert edge_counts == intersection_counts
