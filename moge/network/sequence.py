from abc import ABCMeta, abstractmethod
from typing import Dict, Union, Iterable, List

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.data import get_tokenizer
from torchtext.vocab import Vocab, build_vocab_from_iterator
from transformers import AutoTokenizer


class Sequence(metaclass=ABCMeta):
    def __init__(self, nodes: Dict[str, pd.Series], node_types: List):
        self.nodes = nodes
        self.node_types = node_types

    @abstractmethod
    def build_vocab(self, sequences: Union[Iterable, pd.Series], node_type: str):
        pass

    @abstractmethod
    def one_hot_encode(self, node_type: str, sequences: Union[Iterable, pd.Series], **kwargs):
        pass


class BertSequenceTokenizer(Sequence):
    def __init__(self, nodes: Dict[str, pd.Series],
                 sequences: Dict[str, pd.Series],
                 tokenizers: Dict[str, AutoTokenizer]):
        super().__init__(nodes, node_types=list(tokenizers.keys()))
        self.tokenizers: Dict[str, AutoTokenizer] = tokenizers
        if sequences is not None:
            self.sequences = sequences

    def one_hot_encode(self, node_type: str, sequences: Union[Iterable, pd.Series], node_ids=None):
        output = self.tokenizers[node_type].batch_encode_plus(sequences.to_list(), add_special_tokens=True,
                                                              padding=True, truncation=True,
                                                              return_tensors="pt")
        return output


class CharTokenizer(Sequence):
    def __init__(self, nodes, node_types: List):
        super().__init__(nodes, node_types)
        self.vocab: Dict[str, Vocab] = {}

    def build_vocab(self, sequences: pd.Series, node_type: str):
        tokenizer = get_tokenizer(lambda word: [char for char in word])

        def yield_tokens(data_iter):
            for text in data_iter:
                yield tokenizer(text)

        if node_type in self.vocab:
            vocab = self.vocab[node_type]
        else:
            vocab = build_vocab_from_iterator(yield_tokens(sequences[sequences.notnull()]))

            # Ensures RNA types with the same vocab have the same word indexing
            for other_ntype, other_vocab in self.vocab.items():
                if set(other_vocab.vocab.itos_) == set(vocab.vocab.itos_):
                    vocab = other_vocab

            vocab.set_default_index(-1)
            self.vocab[node_type] = vocab

        return vocab, tokenizer

    def one_hot_encode(self, node_type, sequences: pd.Series, max_length=0.75):
        if not hasattr(self, "vocab"):
            self.vocab = {}

        vocab, tokenizer = self.build_vocab(sequences, node_type)
        if isinstance(max_length, float):
            max_len = int(sequences.map(len).quantile(max_length))
        else:
            max_len = None

        def encoder(x):
            if max_len is not None and len(x) > max_len:
                x = x[: max_len]
            return torch.tensor(vocab(tokenizer(x)))

        seqs = sequences.apply(encoder).to_list()
        seq_lens = torch.tensor([seq.shape[0] for seq in seqs])

        padded_encoding = pad_sequence(seqs, batch_first=True)
        # packed_seqs = PackedSequence(seqs, batch_sizes=seq_lens)

        return padded_encoding, seq_lens
