import os.path
from pprint import pprint
from typing import Dict, Optional, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import HeteroData
from transformers import AutoTokenizer, BatchEncoding
from transformers import (
    BertTokenizer,
    DataCollatorForLanguageModeling
)

from moge.model.transformers import DNATokenizer


class SequenceTokenizers():
    def __init__(self, vocabularies: Dict[str, str], max_length: Union[int, Dict[str, int]] = None):
        self.tokenizers: Dict[str, BertTokenizer] = {}
        self.word_lengths: Dict[str, int] = {}
        self.max_length: Dict[str, int] = {ntype: max_length \
                                           for ntype in vocabularies} if isinstance(max_length, int) else max_length

        for ntype, vocab_file in vocabularies.items():
            if os.path.exists(vocab_file):
                self.tokenizers[ntype] = DNATokenizer.from_pretrained(vocab_file)
            else:
                self.tokenizers[ntype] = AutoTokenizer.from_pretrained(vocab_file)

            # get most frequent word length
            self.word_lengths[ntype] = pd.Series(self.tokenizers[ntype].vocab.keys()).str.len().mode().item()

        pprint({f"{ntype} tokenizer": f"vocab_size={tokenizer.vocab_size}, "
                                      f"word_length={self.word_lengths[ntype]}, "
                                      f"max_length={self.max_length[ntype]}" \
                for ntype, tokenizer in self.tokenizers.items()})

    def __getitem__(self, item: str):
        return self.tokenizers[item]

    def items(self):
        return self.tokenizers.items()

    def encode_sequences(self, batch: HeteroData, ntype: str, max_length: Optional[int] = None, **kwargs) -> \
            BatchEncoding:
        seqs = batch[ntype].sequence.iloc[batch[ntype].nid]
        match_regex = "." * self.word_lengths[ntype]
        seqs = seqs.str.findall(match_regex).str.join(" ")

        if max_length is None and self.max_length is not None:
            max_length = self.max_length[ntype]

        encodings = self.tokenizers[ntype].batch_encode_plus(seqs.tolist(), padding=True,
                                                             max_length=max_length, truncation=True,
                                                             add_special_tokens=True, return_tensors="pt", **kwargs)
        return encodings


class MaskedLMDataset(Dataset):
    def __init__(self, data: Union[os.PathLike, pd.Series],
                 tokenizer: BertTokenizer, mlm_probability=0.15, max_length=None):
        self.tokenizer = tokenizer
        self.max_len = max_length
        self.mlm_probability = mlm_probability

        if isinstance(data, str):
            self.sequences = self.load_lines(data)
        elif isinstance(data, list):
            self.sequences = data
        elif isinstance(data, pd.Series):
            if not any(" " in seq for seq in data[:10]):
                word_length = pd.Series(tokenizer.vocab.keys()).str.len().mode().item()

                # data = data.map(lambda seq: k_mers(seq, k=word_length))
                data = data.str.findall("." * word_length).str.join(" ")
                print(data.str.len().describe())

            self.sequences = data.tolist()

        self.ids = self.encode_lines(self.sequences)

        num_train_samples = int(0.98 * len(self.ids))
        self.training_idx = torch.arange(0, num_train_samples)
        self.validation_idx = self.testing_idx = torch.arange(num_train_samples, len(self.ids))

        self.data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True,
                                                             mlm_probability=self.mlm_probability)

    def load_lines(self, file):
        with open(file) as f:
            lines = [line for line in f.read().splitlines()
                     if (len(line) > 0 and not line.isspace())]
        return lines

    def encode_lines(self, lines):
        batch_encoding = self.tokenizer.batch_encode_plus(lines, add_special_tokens=True,
                                                          truncation=True, max_length=self.max_len)

        return batch_encoding["input_ids"]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.ids[idx], dtype=torch.long)

    def train_dataloader(self, batch_size=128, num_workers=0, **kwargs):
        loader = DataLoader(self, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers,
                            collate_fn=self.data_collator,
                            **kwargs)
        return loader

    def valid_dataloader(self, batch_size=128, num_workers=0, **kwargs):
        loader = DataLoader(self, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers,
                            collate_fn=self.data_collator,
                            **kwargs)
        return loader

    def test_dataloader(self, batch_size=128, num_workers=0, **kwargs):
        loader = DataLoader(self, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers,
                            collate_fn=self.data_collator,
                            **kwargs)
        return loader


def k_mers(sentence: str, k: int = 3, concat: bool = True):
    substrs = [sentence[i: i + k + 1] for i in range(0, len(sentence) - k + 1)]

    if concat:
        substrs = " ".join(substrs)
    return substrs
