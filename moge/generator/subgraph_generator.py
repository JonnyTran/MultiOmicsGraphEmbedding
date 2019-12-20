from .data_generator import DataGenerator


class SubgraphGenerator(DataGenerator):
    def __init__(self, network, weighted=False, batch_size=1, maxlen=1400, padding='post', truncating='post',
                 sequence_to_matrix=False, tokenizer=None, shuffle=True, seed=0, verbose=True, training_network=None):
        super(SubgraphGenerator, self).__init__(network, weighted, batch_size, maxlen, padding, truncating,
                                                sequence_to_matrix, tokenizer,
                                                shuffle, seed, verbose, training_network)
