import random

import pytest
import torch
from cogdl.datasets.han_data import DBLP_HANDataset
from ogb.nodeproppred import PygNodePropPredDataset

from moge.generator import HeteroNeighborGenerator
from moge.generator.utils import edge_dict_intersection, edge_dict_sizes
from moge.generator.utils import nonduplicate_indices
from moge.module.utils import tensor_sizes

dataset_path = "/home/jonny/Bioinformatics_ExternalData/OGB/"


@pytest.fixture
def generate_dataset_hetero():
    ogbn = PygNodePropPredDataset(name="ogbn-mag", root=dataset_path)
    return ogbn


@pytest.fixture
def generate_dataset_homo():
    return DBLP_HANDataset()


@pytest.fixture
def generator_hetero(generate_dataset_hetero):
    dataset = HeteroNeighborGenerator(generate_dataset_hetero, neighbor_sizes=[20, 10],
                                      head_node_type="paper",
                                      edge_dir=True,
                                      add_reverse_metapaths=True)
    return dataset


@pytest.fixture
def generator_homo(generate_dataset_homo):
    dataset = HeteroNeighborGenerator(DBLP_HANDataset(), neighbor_sizes=[25, 20],
                                      node_types=["A", "P", "C", "T"], head_node_type="A",
                                      metapaths=["AC", "AP", "AT"],
                                      add_reverse_metapaths=True, inductive=True)
    dataset.x_dict["P"] = dataset.x_dict["A"]
    dataset.x_dict["C"] = dataset.x_dict["A"]
    dataset.x_dict["T"] = dataset.x_dict["A"]
    return dataset


def test_generator_hetero(generator_hetero):
    X, y, weights = generator_hetero.sample(random.sample(generator_hetero.training_idx.numpy().tolist(), 50),
                                            mode="train")
    print(tensor_sizes({"X": X, "y": y}))
    assert X is not None
    assert y is not None

    assert y.size(0) == X["global_node_index"][generator_hetero.head_node_type].size(0)
    assert y.size(0) == weights.size(0)

    X, y, weights = generator_hetero.sample(random.sample(generator_hetero.validation_idx.numpy().tolist(), 50),
                                            mode="valid")
    print(tensor_sizes({"X": X, "y": y}))
    assert X is not None
    assert y is not None


def test_generator_homo(generator_homo):
    X, y, weights = generator_homo.sample(random.sample(generator_homo.training_idx.numpy().tolist(), 50), mode="train")
    print(tensor_sizes({"X": X, "y": y}))
    assert X is not None
    assert y is not None

    assert y.size(0) == X["global_node_index"][generator_homo.head_node_type].size(0)
    assert y.size(0) == weights.size(0)

    X, y, weights = generator_homo.sample(random.sample(generator_homo.validation_idx.numpy().tolist(), 50),
                                          mode="valid")
    print(tensor_sizes({"X": X, "y": y}))
    assert X is not None
    assert y is not None


def test_sampled_edges_exists_hetero(generator_hetero):
    node_idx = random.sample(generator_hetero.training_idx.numpy().tolist(), 50)
    batch_size, n_id, adjs = generator_hetero.graph_sampler.sample(node_idx)

    global_node_index = generator_hetero.graph_sampler.get_nodes_dict(adjs, n_id, )

    edge_index = generator_hetero.graph_sampler.get_edge_index_dict(adjs, n_id, global_node_index,
                                                                    filter_nodes=False)

    edge_index = {k: torch.stack([global_node_index[k[0]][v[0]], global_node_index[k[-1]][v[1]]], axis=0) \
                  for k, v in edge_index.items()}

    edge_counts = edge_dict_sizes({k: eids[:, nonduplicate_indices(eids)] for k, eids in edge_index.items()})
    intersection_counts = edge_dict_sizes(edge_dict_intersection(edge_index, generator_hetero.edge_index_dict))

    assert edge_counts == intersection_counts


def test_sampled_edges_exists_homo(generator_homo):
    node_idx = random.sample(generator_homo.training_idx.numpy().tolist(), 50)
    batch_size, n_id, adjs = generator_homo.graph_sampler.sample(node_idx)

    global_node_index = generator_homo.graph_sampler.get_nodes_dict(adjs, n_id, )

    edge_index = generator_homo.graph_sampler.get_edge_index_dict(adjs, n_id, global_node_index,
                                                                  filter_nodes=False)

    edge_index = {k: torch.stack([global_node_index[k[0]][v[0]], global_node_index[k[-1]][v[1]]], axis=0) \
                  for k, v in edge_index.items()}

    edge_counts = edge_dict_sizes({k: eids[:, nonduplicate_indices(eids)] for k, eids in edge_index.items()})
    intersection_counts = edge_dict_sizes(edge_dict_intersection(edge_index, generator_homo.edge_index_dict))

    assert edge_counts == intersection_counts
