from typing import Dict, Optional

import pandas as pd
from torch_geometric.data import HeteroData

from moge.model.transformers import DNATokenizer


class HeteroSequence():
    def __init__(self, vocabularies: Dict[str, str], max_length: Dict[str, int] = None):
        self.tokenizers = {}
        self.word_lengths = {}
        self.max_length = max_length

        for ntype, vocab_file in vocabularies.items():
            self.tokenizers[ntype] = DNATokenizer.from_pretrained(vocab_file)
            # get most frequent word length
            self.word_lengths[ntype] = pd.Series(self.tokenizers[ntype].vocab.keys()).str.len().mode().item()

        print("Vocab word lengths", self.word_lengths)

    def encode_sequences(self, batch: HeteroData, ntype: str, max_length: Optional[int] = None):
        assert hasattr(batch, "sequence_dict")
        seqs = batch[ntype].sequence.iloc[batch[ntype].nid]
        seqs = seqs.str.findall("...").str.join(" ")

        encoding = self.tokenizers[ntype].batch_encode_plus(seqs, add_special_tokens=True, return_tensors="pt",
                                                            padding='longest',
                                                            max_length=max_length)
        return encoding
