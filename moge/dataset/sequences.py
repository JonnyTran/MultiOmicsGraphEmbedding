import os.path
from pprint import pprint
from typing import Dict, Optional, Union, List

import pandas as pd
from torch_geometric.data import HeteroData
from transformers import AutoTokenizer, BertTokenizer, BatchEncoding

from moge.model.transformers import DNATokenizer


class SequenceTokenizer():
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
        seqs = seqs.str.findall("." * self.word_lengths[ntype]).str.join(" ")

        if max_length is None and self.max_length is not None:
            max_length = self.max_length[ntype]

        encodings = self.tokenizers[ntype].batch_encode_plus(seqs.tolist(), padding='longest', max_length=max_length,
                                                             add_special_tokens=True, return_tensors="pt", **kwargs)
        return encodings


def kmers(s: str, k: int = 3, concat: bool = True) -> Union[List[str], str]:
    res = [s[i: j] \
           for i in range(len(s)) \
           for j in range(i + 1, len(s) + 1) if len(s[i:j]) == k]

    if concat:
        res = " ".join(res)
    return res
