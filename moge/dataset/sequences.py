from typing import Dict, Optional

import pandas as pd
from torch import Tensor
from torch_geometric.data import HeteroData

from moge.model.transformers import DNATokenizer


class SequenceTokenizer():
    def __init__(self, vocabularies: Dict[str, str], max_length: Dict[str, int] = None):
        self.tokenizers: Dict[str, DNATokenizer] = {}
        self.word_lengths: Dict[str, int] = {}
        self.max_length: Dict[str, int] = max_length

        for ntype, vocab_file in vocabularies.items():
            self.tokenizers[ntype] = DNATokenizer.from_pretrained(vocab_file)
            # get most frequent word length
            self.word_lengths[ntype] = pd.Series(self.tokenizers[ntype].vocab.keys()).str.len().mode().item()

        print("Vocab word lengths", self.word_lengths)

    def encode_sequences(self, batch: HeteroData, ntype: str, max_length: Optional[int] = None, **kwargs) -> Dict[
        str, Tensor]:
        assert hasattr(batch, "sequence_dict")
        seqs = batch[ntype].sequence.iloc[batch[ntype].nid]
        seqs = seqs.str.findall("...").str.join(" ")

        if max_length is None and self.max_length is not None:
            max_length = self.max_length

        encodings = self.tokenizers[ntype].batch_encode_plus(seqs, padding='longest', max_length=max_length,
                                                             add_special_tokens=True, return_tensors="pt", )
        return encodings
