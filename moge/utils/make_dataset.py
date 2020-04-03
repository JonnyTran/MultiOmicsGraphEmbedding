import random

from moge.generator.subgraph_generator import SubgraphGenerator


def make_dataset(network, directed=False,
                 targets=['go_id'],
                 test_frac=0.15,
                 max_length=1000, batch_size=1500, seed=None, verbose=True):
    variables = []
    n_steps = int(400000 / batch_size)
    if seed is None:
        seed = random.randint(0, 10000)

    network.split_stratified(directed=directed, stratify_label=targets[0], stratify_label_2=True,
                             n_splits=int(1 / test_frac),
                             dropna=False, seed=seed, verbose=verbose)

    generator_train = network.get_train_generator(
        SubgraphGenerator, variables=variables, targets=targets,
        batch_size=batch_size,
        compression_func="linear", n_steps=n_steps, directed=directed,
        maxlen=max_length, padding='post', truncating='post',
        seed=seed, verbose=verbose)

    test_batch_size = batch_size
    test_n_steps = max(int(len(network.testing.node_list) / test_batch_size), 1)

    generator_test = network.get_test_generator(
        SubgraphGenerator, variables=variables, targets=targets,
        batch_size=test_batch_size,
        compression_func="linear", n_steps=test_n_steps, directed=directed,
        maxlen=max_length, padding='post', truncating='post',
        seed=seed, verbose=verbose)

    assert generator_train.tokenizer.word_index == generator_test.tokenizer.word_index
    return generator_train, generator_test
