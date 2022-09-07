import itertools
from abc import ABC
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import scipy.sparse as ssp
import torch
from colorhash import ColorHash
from pandas import DataFrame
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GATv2Conv
from torch_sparse import SparseTensor
from torchtyping import TensorType

from moge.model.PyG.utils import filter_metapaths, get_edge_index_values


class MetapathGATConv(nn.Module):
    def __init__(self, embedding_dim: int, metapaths: List[Tuple[str, str, str]], n_layers=2,
                 attn_heads=4, attn_dropout=0.0):
        super().__init__()
        self.metapaths = metapaths
        self.n_relations = len(metapaths) + 1
        self.self_index = self.n_relations - 1

        self.edge_indexes = {n: self.generate_fc_edge_index(num_src_nodes=n) for n in range(1, self.n_relations + 1)}

        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.out_channels = embedding_dim // attn_heads

        self.layers: List[GATConv] = nn.ModuleList([
            GATv2Conv(in_channels=embedding_dim, out_channels=self.out_channels, add_self_loops=False,
                      heads=attn_heads, dropout=attn_dropout) \
            for _ in range(n_layers)
        ])
        # self.norm = GraphNorm(embedding_dim)

    def generate_fc_edge_index(self, num_src_nodes: int, num_dst_nodes: int = None, device=None):
        if num_dst_nodes is None:
            num_dst_nodes = num_src_nodes

        edge_index = torch.tensor(list(itertools.product(range(num_src_nodes), range(num_dst_nodes))),
                                  device=device, dtype=torch.long, requires_grad=False).T
        return edge_index

    def construct_multigraph(self, relation_embs: TensorType["num_nodes", "n_relations", "embedding_dim"]) \
            -> Data:
        num_nodes = relation_embs.size(0)
        nid = torch.arange(self.n_relations, device=relation_embs.device)

        data_list = []
        for i in torch.arange(num_nodes):
            x = relation_embs[i]
            node_mask = torch.count_nonzero(x, dim=1).type(torch.bool)
            num_nz_relations = node_mask.sum().item()

            g = Data(x=x[node_mask], nid=nid[node_mask],
                     edge_index=self.edge_indexes[num_nz_relations].to(relation_embs.device))
            data_list.append(g)

        loader = DataLoader(data_list, batch_size=len(data_list), shuffle=False)
        batch: Data = next(iter(loader))

        return batch

    def deconstruct_multigraph(self, batch: Data,
                               h: TensorType["batch_nodes", "embedding_dim"],
                               alpha_edges: TensorType[2, "batch_edges"],
                               alpha_values: TensorType["batch_edges", "attn_heads"]):
        node_embs = h[batch.nid == self.self_index]

        if alpha_edges is not None and alpha_values is not None:
            src_node_id = batch.nid[alpha_edges[0]]
            dst_node_id = batch.nid[alpha_edges[1]]
            batch_id = batch.batch[alpha_edges[1]]

            self_node_mask = dst_node_id == self.self_index

            betas = SparseTensor(row=batch_id[self_node_mask],
                                 col=src_node_id[self_node_mask],
                                 value=alpha_values[self_node_mask]).to_dense().detach()
        else:
            betas = None

        return node_embs, betas

    def forward(self, relation_embs: TensorType["num_nodes", "n_relations", "embedding_dim"]):
        batch: Data = self.construct_multigraph(relation_embs)

        h = torch.relu(batch.x)

        for i in range(self.n_layers):
            is_last_layer = i + 1 == self.n_layers

            if is_last_layer:
                # Select only edges with dst as the readout node
                edge_mask = batch.nid[batch.edge_index[1]] == self.self_index
                edge_index = batch.edge_index[:, edge_mask]
            else:
                edge_index = batch.edge_index

            h, (alpha_edges, alpha_values) = self.layers[i].forward(h, edge_index, return_attention_weights=True)

            h = torch.relu(h)
            if hasattr(self, 'norm'):
                h = self.norm(h)

        node_embs, betas = self.deconstruct_multigraph(batch, h, alpha_edges, alpha_values)
        return node_embs, betas


class RelationAttention(ABC):
    metapaths: List[Tuple[str, str, str]]
    _betas: Dict[str, DataFrame]
    _alphas: Dict[str, DataFrame]
    _counts: Dict[str, DataFrame]

    def __init__(self):
        self.reset_betas()

    def reset_betas(self):
        self._counts = {}
        self._betas = {}
        self._alphas = {}

    def get_src_ntypes(self, metapaths=None):
        if metapaths is None:
            metapaths = self.metapaths
        return {metapath[0] for metapath in metapaths}

    def get_dst_ntypes(self, metapaths=None):
        if metapaths is None:
            metapaths = self.metapaths
        return {metapath[-1] for metapath in metapaths}

    def get_head_relations(self, src_node_type, order=None, str_form=False) -> List[Tuple[str, str, str]]:
        relations = filter_metapaths(self.metapaths, order=order, head_type=src_node_type)

        if str_form:
            relations = [".".join(metapath) if isinstance(metapath, tuple) else metapath \
                         for metapath in relations]

        return relations

    def get_tail_relations(self, dst_node_type, order=None, str_form=False) -> List[Tuple[str, str, str]]:
        relations = filter_metapaths(self.metapaths, order=order, tail_type=dst_node_type)

        if str_form:
            relations = [".".join(metapath) if isinstance(metapath, tuple) else metapath \
                         for metapath in relations]
        return relations

    def num_head_relations(self, node_type) -> int:
        """
        Return the number of metapaths with head node type equals to :param ntype: and plus one for none-selection.
        """
        relations = self.get_head_relations(node_type)
        return len(relations) + 1

    def num_tail_relations(self, ntype) -> int:
        relations = self.get_tail_relations(ntype)
        return len(relations) + 1

    @torch.no_grad()
    def update_edge_attn(self, edge_index_dict: Dict[Tuple[str, str, str], Tensor],
                         global_node_index: Dict[str, Tensor],
                         batch_sizes: Dict[str, int] = None, save_count_only=False):
        if not hasattr(self, "_counts"):
            self._counts = {}
        if not hasattr(self, "_alphas"):
            self._alphas = {}

        for ntype, nids in global_node_index.items():
            if batch_sizes != None and ntype not in batch_sizes:
                continue
            elif batch_sizes[ntype] == 0:
                continue
            batch_nids = nids[:batch_sizes[ntype]].cpu().numpy()
            counts_df = []

            for metapath in filter_metapaths(edge_index_dict, tail_type=ntype):
                metapath_name = ".".join(metapath)
                head_type, tail_type = metapath[0], metapath[-1]
                edge_index, edge_values = get_edge_index_values(edge_index_dict[metapath], )

                dst_nids, dst_edge_counts = edge_index[1].cpu().unique(return_counts=True)
                dst_nids = nids[dst_nids].cpu().numpy()

                # Edge counts
                dst_counts = pd.concat([
                    pd.Series(dst_edge_counts.cpu().numpy(), index=dst_nids, name=metapath_name, dtype=int),
                    pd.Series(0, index=set(batch_nids).difference(dst_nids), name=metapath_name, dtype=int)])
                dst_counts = dst_counts.loc[batch_nids]
                counts_df.append(dst_counts)

                # Edge attn
                if edge_values is None or save_count_only or tail_type not in batch_sizes: continue
                value, row, col = edge_values.cpu().max(1).values.numpy(), \
                                  edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
                csc_matrix = ssp.coo_matrix((value, (row, col)),
                                            shape=(global_node_index[head_type].shape[0],
                                                   global_node_index[tail_type].shape[0])).transpose().tocsc()
                edge_attn = pd.DataFrame.sparse.from_spmatrix(csc_matrix)
                edge_attn.index = pd.Index(global_node_index[tail_type].cpu().numpy(), name=f"{tail_type}_nid")
                edge_attn.columns = pd.Index(global_node_index[head_type].cpu().numpy(), name=f"{head_type}_nid")
                edge_attn = edge_attn.loc[batch_nids]

                if len(self._alphas) == 0 or metapath_name not in self._alphas:
                    self._alphas[metapath_name] = edge_attn
                else:
                    existing_cols = edge_attn.columns.intersection(self._alphas[metapath_name].columns)
                    new_cols = edge_attn.columns.difference(self._alphas[metapath_name].columns)

                    if len(new_cols):
                        self._alphas[metapath_name] = self._alphas[metapath_name].join(
                            edge_attn.filter(new_cols, axis="columns"), how="left")
                    # Fillna seq features
                    if len(existing_cols):
                        self._alphas[metapath_name].update(
                            edge_attn.filter(existing_cols, axis='columns'), overwrite=True)


            if counts_df:
                counts_df = pd.concat(counts_df, axis=1, join="outer", copy=False) \
                    .fillna(0).astype(int, copy=False)
                counts_df.index.name = f"{ntype}_nid"

                if len(self._counts) == 0 or ntype not in self._counts:
                    self._counts[ntype] = counts_df
                else:
                    self._counts[ntype].update(counts_df, overwrite=True)

    @torch.no_grad()
    def update_relation_attn(self, betas: Dict[str, Tensor],
                             global_node_index: Dict[str, Tensor],
                             batch_size: Dict[str, int]):
        # Only save relation weights if beta has weights for all node_types in the global_node_idx batch
        if len(betas) < len({metapath[-1] for metapath in self.metapaths}):
            return

        if not hasattr(self, "_betas"):
            self._betas = {}

        for ntype in betas:
            if ntype not in global_node_index or global_node_index[ntype].numel() == 0: continue
            if batch_size and ntype not in batch_size:
                continue
            elif batch_size[ntype] == 0:
                continue

            relations = self.get_tail_relations(ntype, str_form=True) + [ntype, ]
            if len(relations) <= 1: continue

            nids = global_node_index[ntype].cpu().numpy()
            batch_nids = nids[:batch_size[ntype]]

            df = pd.DataFrame(betas[ntype].squeeze(-1).cpu().numpy(),
                              columns=relations, index=nids, dtype=np.float16)
            df = df.loc[batch_nids]
            df.index.name = f"{ntype}_nid"

            if len(self._betas) == 0 or ntype not in self._betas:
                self._betas[ntype] = df
            else:
                self._betas[ntype].update(df, overwrite=True)

    @property
    def _beta_std(self):
        if hasattr(self, "_betas"):
            return {ntype: betas.std(0).to_dict() for ntype, betas in self._betas.items()}

    @property
    def _beta_avg(self):
        if hasattr(self, "_betas"):
            return {ntype: betas.mean(0).to_dict() for ntype, betas in self._betas.items()}

    def get_relation_weights(self, std=True) -> Dict[str, Tuple[float, float]]:
        """
        Get the mean and std of relation attention weights for all nodes
        """
        output = {}
        for node_type in self._beta_avg:
            for metapath, avg in self._beta_avg[node_type].items():
                if std:
                    output[metapath] = (avg, self._beta_std[node_type][metapath])
                else:
                    output[metapath] = avg
        return output

    def get_sankey_flow(self, node_type: str, self_loop: bool = False, agg="median") \
            -> Tuple[DataFrame, DataFrame]:

        rel_attn = self._betas[node_type]

        if agg == "sum":
            rel_attn_agg = rel_attn.sum(axis=0)
        elif agg == "median":
            rel_attn_agg = rel_attn.median(axis=0)
        elif agg == "max":
            rel_attn_agg = rel_attn.max(axis=0)
        elif agg == "min":
            rel_attn_agg = rel_attn.min(axis=0)
        else:
            rel_attn_agg = rel_attn.mean(axis=0)

        rel_attn_std = rel_attn.std(axis=0)

        # Break down each metapath tuples into nodes
        indexed_metapaths = rel_attn_agg.index.str.split(".").map(
            lambda tup: [str(len(tup) - i) + name for i, name in enumerate(tup)])

        indexed_nodes = {node for nodes in indexed_metapaths for node in nodes}
        indexed_nodes = {node: i for i, node in enumerate(indexed_nodes)}
        indexed_node2metapath = {node: ".".join([s[1:] for s in nodes_tup]) \
                                 for nodes_tup in indexed_metapaths for node in nodes_tup}

        # Links
        links = pd.DataFrame(columns=["source", "target", "mean", "std", "label", "color"])
        for i, (metapath_name, attn_agg) in enumerate(rel_attn_agg.to_dict().items()):
            indexed_metapath = indexed_metapaths[i]

            if len(metapath_name.split(".")) >= 2:
                sources = [indexed_nodes[indexed_metapath[j]] for j, _ in enumerate(indexed_metapath[:-1])]
                targets = [indexed_nodes[indexed_metapath[j + 1]] for j, _ in enumerate(indexed_metapath[:-1])]

                path_links = pd.DataFrame({"source": sources,
                                           "target": targets,
                                           "label": [metapath_name, ] * len(targets),
                                           "mean": [attn_agg, ] * len(targets),
                                           'std': [rel_attn_std.loc[metapath_name], ] * len(targets),
                                           })
                links = links.append(path_links, ignore_index=True)


            elif self_loop:
                source = indexed_nodes[indexed_metapath[0]]
                links = links.append({"source": source, "target": source, "label": metapath_name,
                                      "mean": attn_agg, "std": rel_attn_std.loc[node_type], }, ignore_index=True)

        links["color"] = links["label"].apply(lambda label: ColorHash(label).hex)
        links = links.iloc[::-1]

        # Nodes
        # node_group = [int(node[0]) for node, nid in all_nodes.items()]
        # groups = [[nid for nid, node in enumerate(node_group) if node == group] for group in np.unique(node_group)]

        nodes = pd.DataFrame(columns=["label", "metapath", "level", "color", "count"])
        nodes["label"] = [node[1:] for node in indexed_nodes.keys()]
        nodes["level"] = [int(node[0]) for node in indexed_nodes.keys()]
        nodes["metapath"] = [indexed_node2metapath[node] for node in indexed_nodes.keys()]

        # Get number of edge_index counts for each metapath
        if node_type in self._counts:
            nodes["count"] = nodes["metapath"].map(lambda m: self._counts[node_type].sum(axis=0).get(m, None))
        # Set count of target nodes
        nodes.loc[nodes.query(f'label == "{node_type}" & metapath == "{node_type}"').index, 'count'] = rel_attn.shape[0]

        nodes["color"] = nodes[["label", "level"]].apply(
            lambda x: ColorHash(x["label"].strip("rev_")).hex \
                if x["level"] % 2 == 0 \
                else ColorHash(x["label"]).hex,
            axis=1)

        return nodes, links
