from typing import List, Tuple, Union, Dict, Optional

import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from moge.dataset.PyG.neighbor_sampler import HGTLoader
from moge.dataset.graph import HeteroGraphDataset
from moge.model.transformers.tokenizers import DNATokenizer


# from torch_geometric.loader import HGTLoader, NeighborLoader

class HeteroDataSampler(HeteroGraphDataset):
    def __init__(self, dataset: HeteroData, vocabularies: Dict[str, str] = None, max_length: Dict[str, int] = None,
                 neighbor_sizes: Union[List[int], Dict[str, List[int]]] = [128, 128],
                 node_types: List[str] = None, metapaths: List[Tuple[str, str, str]] = None, head_node_type: str = None,
                 edge_dir: str = "in", reshuffle_train: float = None, add_reverse_metapaths: bool = True,
                 inductive: bool = False, **kwargs):
        super().__init__(dataset, node_types, metapaths, head_node_type, edge_dir, reshuffle_train,
                         add_reverse_metapaths, inductive, **kwargs)

        self.neighbor_sizes = neighbor_sizes
        if vocabularies:
            self.process_sequences(vocabularies, max_length)

    def process_pyg_heterodata(self, hetero: HeteroData):
        self.G = hetero
        self.x_dict = hetero.x_dict
        self.node_types = hetero.node_types
        self.num_nodes_dict = {ntype: hetero[ntype].num_nodes for ntype in hetero.node_types}
        self.y_dict = {ntype: hetero[ntype].y for ntype in hetero.node_types}

        self.metapaths = hetero.edge_types
        self.edge_index_dict = {etype: edge_index for etype, edge_index in zip(hetero.edge_types, hetero.edge_stores)}

    def process_sequences(self, vocabularies: Dict[str, str], max_length: Dict[str, int] = None):
        self.tokenizers = {}
        self.word_lengths = {}
        self.max_length = max_length

        for ntype, vocab_file in vocabularies.items():
            self.tokenizers[ntype] = DNATokenizer.from_pretrained(vocab_file)
            self.word_lengths[ntype] = pd.Series(self.tokenizers[ntype].vocab.keys()).str.len().mode().item()

        print("Vocab word lengths", self.word_lengths)

    @classmethod
    def from_pyg_heterodata(cls, hetero: HeteroData,
                            classes: List[str],
                            train_idx: Dict[str, Tensor],
                            val_idx: Dict[str, Tensor],
                            test_idx: Dict[str, Tensor], **kwargs):
        self = cls(dataset=hetero, metapaths=hetero.edge_types, add_reverse_metapaths=False,
                   edge_dir="in", **kwargs)
        self.classes = classes
        self._name = ""

        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx

        return self

    def sample(self, batch: HeteroData):
        X = {}
        X["x_dict"] = {ntype: x for ntype, x in batch.x_dict.items() if x.size(0)}
        X["edge_index_dict"] = batch.edge_index_dict
        X["global_node_index"] = {ntype: nid for ntype, nid in batch.nid_dict.items() if nid.numel()}
        X['sizes'] = {ntype: size for ntype, size in batch.num_nodes_dict.items() if size}
        X['batch_size'] = batch.batch_size_dict

        if hasattr(self, "tokenizers"):
            X["sequences"] = {}
            for ntype in X["global_node_index"]:
                X["sequences"][ntype] = self.encode_sequences(batch, ntype,
                                                              max_length=self.max_length[ntype] if isinstance(
                                                                  self.max_length, dict) else None)

        y_dict = {ntype: y for ntype, y in batch.y_dict.items() if y.size(0)}

        if len(y_dict) == 1:
            y_dict = y_dict[list(y_dict.keys()).pop()]

            if y_dict.dim() == 2 and y_dict.size(1) == 1:
                y_dict = y_dict.squeeze(-1)
            elif y_dict.dim() == 1:
                weights = (y_dict >= 0).to(torch.float)

        elif len(y_dict) > 1:
            weights = {}
            for ntype, label in y_dict.items():
                if label.dim() == 2 and label.size(1) == 1:
                    y_dict[ntype] = label.squeeze(-1)

                if label.dim() == 1:
                    weights[ntype] = (y_dict >= 0).to(torch.float)
                elif label.dim() == 2:
                    weights[ntype] = (label.sum(1) > 0).to(torch.float)

        return X, y_dict, weights

    def encode_sequences(self, batch: HeteroData, ntype: str, max_length: Optional[int] = None):
        seqs = batch[ntype].sequence.iloc[batch[ntype].nid]
        seqs = seqs.str.findall("...").str.join(" ")

        encoding = self.tokenizers[ntype].batch_encode_plus(seqs, add_special_tokens=True, return_tensors="pt",
                                                            padding='longest',
                                                            max_length=max_length)
        return encoding

    def train_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, **kwargs):
        dataset = HGTLoader(self.G, num_neighbors=self.neighbor_sizes,
                            batch_size=batch_size,
                            # directed=True,
                            transform=self.sample,
                            input_nodes=(self.head_node_type, self.G[self.head_node_type].train_mask),
                            shuffle=True,
                            num_workers=num_workers,
                            **kwargs)

        return dataset

    def valid_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, **kwargs):
        dataset = HGTLoader(self.G, num_neighbors=self.neighbor_sizes,
                            batch_size=batch_size,
                            # directed=False,
                            transform=self.sample,
                            input_nodes=(self.head_node_type, self.G[self.head_node_type].valid_mask),
                            shuffle=False, num_workers=num_workers, **kwargs)

        return dataset

    def test_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, **kwargs):
        dataset = HGTLoader(self.G, num_neighbors=self.neighbor_sizes,
                            batch_size=batch_size,
                            # directed=False,
                            transform=self.sample,
                            input_nodes=(self.head_node_type, self.G[self.head_node_type].test_mask),
                            shuffle=False, num_workers=num_workers, **kwargs)

        return dataset

