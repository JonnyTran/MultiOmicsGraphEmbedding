import pickle

import pytest

from moge.generator import SubgraphGenerator
from moge.network import HeterogeneousNetwork

cohort_folder_path = "tests/data/luad_data_openomics.pickle"


@pytest.fixture
def get_luad_data_network() -> HeterogeneousNetwork:
    with open(cohort_folder_path, 'rb') as file:
        luad_data = pickle.load(file)

    network = HeterogeneousNetwork(modalities=["MicroRNA", "MessengerRNA", "LncRNA"],
                                   multi_omics_data=luad_data)
    network.import_edgelist_file(
        file="moge/data/LMN_future_recall/TRAIN/Interactions_Only/GE/lmn_train.BioGRID.interactions.edgelist",
        is_directed=True)

    network.import_edgelist_file(
        file="moge/data/LMN_future_recall/TRAIN/Interactions_Only/MIR/lmn_train.miRTarBase.interactions.edgelist",
        is_directed=True)

    network.import_edgelist_file(
        file="moge/data/LMN_future_recall/TRAIN/Interactions_Only/LNC/lmn_train.lncBase.interactions.edgelist",
        is_directed=True)

    network.import_edgelist_file(
        file="moge/data/LMN_future_recall/TRAIN/Interactions_Only/LNC/lmn_train.lncrna2target.interactions.edgelist",
        is_directed=True)
    return network


@pytest.fixture
def get_traintestsplit_network(get_luad_data_network):
    get_luad_data_network.split_train_test_nodes(get_luad_data_network.node_list, verbose=False)
    assert hasattr(get_luad_data_network, 'train_network')
    assert hasattr(get_luad_data_network, 'test_network')
    assert hasattr(get_luad_data_network, 'val_network')
    return get_luad_data_network


@pytest.fixture
def get_training_generator(get_traintestsplit_network) -> SubgraphGenerator:
    variables = ['chromosome_name', 'transcript_start', 'transcript_end']
    targets = ['gene_biotype']
    return get_traintestsplit_network.get_train_generator(SubgraphGenerator, variables=variables, targets=targets,
                                                          weighted=False, batch_size=100,
                                                          compression_func="log", n_steps=100, directed_proba=1.0,
                                                          maxlen=1400, padding='post', truncating='post',
                                                          sequence_to_matrix=False, tokenizer=None, replace=False,
                                                          seed=0, verbose=True)


def test_training_generator(get_training_generator):
    X, y = get_training_generator.__getitem__(0)
    print({k: v.shape for k, v in X.items()}, {"y": y.shape})
    print("get_training_generator.variables", get_training_generator.variables)

    assert set(get_training_generator.variables) < set(X.keys())
    for variable in get_training_generator.variables:
        assert X[variable].shape[0] == y.shape[0]


@pytest.fixture
def get_testing_generator(get_traintestsplit_network) -> SubgraphGenerator:
    variables = ['chromosome_name', 'transcript_start', 'transcript_end']
    targets = ['gene_biotype', 'transcript_biotype']
    return get_traintestsplit_network.get_test_generator(SubgraphGenerator, variables=variables, targets=targets,
                                                         weighted=False, batch_size=100,
                                                         compression_func="log", n_steps=100, directed_proba=1.0,
                                                         maxlen=1400, padding='post', truncating='post',
                                                         sequence_to_matrix=False, tokenizer=None, replace=False,
                                                         seed=0, verbose=True)


def test_testing_generator(get_testing_generator):
    X, y = get_testing_generator.__getitem__(0)
    print({k: v.shape for k, v in X.items()}, {k: v.shape for k, v in y.items()})
    print("get_training_generator.variables", get_testing_generator.variables)

    assert set(get_testing_generator.variables) < set(X.keys())
    assert set(get_testing_generator.targets) < set(y.keys())
    for variable in get_testing_generator.variables:
        for target in get_testing_generator.targets:
            assert X[variable].shape[0] == y[target].shape[0]
