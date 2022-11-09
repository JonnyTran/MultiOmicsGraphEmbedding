import itertools
from abc import ABC
from typing import Tuple, List, Dict, Union

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

from moge.model.PyG.utils import filter_metapaths, get_edge_index_values, max_num_hops


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

    def construct_multigraph(self, relation_embs: Tensor) \
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
                               h: Tensor,
                               alpha_edges: Tensor,
                               alpha_values: Tensor):
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

    def forward(self, relation_embs: Tensor):
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
        self.reset()

    def reset(self):
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
    def update_relation_attn(self, betas: Dict[str, Tensor],
                             global_node_index: Dict[str, Tensor],
                             batch_sizes: Dict[str, int]):
        # Only save relation weights if beta has weights for all node_types in the global_node_idx batch
        if not hasattr(self, "_betas"):
            self._betas = {}

        for ntype in betas:
            if ntype not in global_node_index or global_node_index[ntype].numel() == 0: continue

            relations = self.get_tail_relations(ntype, str_form=True) + [ntype, ]
            if len(relations) <= 1: continue

            nids = global_node_index[ntype].cpu().numpy()
            if batch_sizes and ntype in batch_sizes:
                batch_nids = nids[:batch_sizes[ntype]]
            else:
                batch_nids = nids

            betas_mean = betas[ntype].squeeze(-1).cpu().numpy()

            df = pd.DataFrame(betas_mean, columns=relations, index=nids, dtype=np.float16)
            df = df.loc[batch_nids]
            df.index.name = f"{ntype}_nid"

            if len(self._betas) == 0 or ntype not in self._betas:
                self._betas[ntype] = df
            else:
                self._betas[ntype].update(df, overwrite=True)

    @torch.no_grad()
    def update_edge_attn(self, edge_index_dict: Dict[Tuple[str, str, str], Tensor],
                         global_node_index: Dict[str, Tensor],
                         batch_sizes: Dict[str, int] = None,
                         save_count_only=False):
        """
        Save the edge-level attention values and the edge count for each node.

        Args:
            edge_index_dict ():
            global_node_index ():
            batch_sizes ():
            save_count_only ():
        """
        if not hasattr(self, "_counts"):
            self._counts = {}
        if not hasattr(self, "_alphas"):
            self._alphas = {}

        for ntype, nids in global_node_index.items():
            if batch_sizes and ntype in batch_sizes:
                batch_nids = nids[:batch_sizes[ntype]].cpu().numpy()
            else:
                batch_nids = nids.cpu().numpy()
            counts_df = []

            for metapath in filter_metapaths(edge_index_dict, tail_type=ntype):
                metapath_name = ".".join(metapath)
                head_type, tail_type = metapath[0], metapath[-1]
                edge_index, edge_values = get_edge_index_values(edge_index_dict[metapath], )
                if edge_values is None or (batch_sizes and tail_type not in batch_sizes): continue

                # Edge attn
                value, row, col = edge_values.mean(1).cpu().numpy(), \
                                  edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
                csc_matrix = ssp.coo_matrix((value, (row, col)),
                                            shape=(global_node_index[head_type].shape[0],
                                                   global_node_index[tail_type].shape[0]))
                # Create Sparse DataFrame of size (batch_nids, neighbor_nids)
                edge_attn = pd.DataFrame.sparse.from_spmatrix(csc_matrix.transpose().tocsc())
                edge_attn.index = pd.Index(global_node_index[tail_type].cpu().numpy(), name=f"{tail_type}_nid")
                edge_attn.columns = pd.Index(global_node_index[head_type].cpu().numpy(), name=f"{head_type}_nid")
                edge_attn = edge_attn.loc[batch_nids]

                if len(self._alphas) == 0 or metapath_name not in self._alphas:
                    self._alphas[metapath_name] = edge_attn
                else:
                    # Update the df
                    old_cols = edge_attn.columns.intersection(self._alphas[metapath_name].columns)
                    if len(new_cols):
                        self._alphas[metapath_name] = self._alphas[metapath_name].join(
                            edge_attn.filter(new_cols, axis="columns"), how="left")

                    # Fillna attn values
                    new_cols = edge_attn.columns.difference(self._alphas[metapath_name].columns)
                    if len(old_cols):
                        self._alphas[metapath_name].update(
                            edge_attn.filter(old_cols, axis='columns'), overwrite=True)

                # Edge counts
                dst_nids, dst_edge_counts = edge_index[1].cpu().unique(return_counts=True)
                dst_nids = nids[dst_nids].cpu().numpy()

                dst_counts = pd.concat(
                    [pd.Series(dst_edge_counts.cpu().numpy(), index=dst_nids, name=metapath_name, dtype=int),
                     pd.Series(0, index=set(batch_nids).difference(dst_nids), name=metapath_name, dtype=int)],
                    axis=0)
                dst_counts = dst_counts.loc[batch_nids]
                counts_df.append(dst_counts)

            if counts_df:
                counts_df = pd.concat(counts_df, axis=1, join="outer", copy=False) \
                    .fillna(0).astype(int, copy=False)
                counts_df.index.name = f"{ntype}_nid"

                if len(self._counts) == 0 or ntype not in self._counts:
                    self._counts[ntype] = counts_df
                else:
                    self._counts[ntype].update(counts_df, overwrite=True)

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

    def get_sankey_flow(self, node_types: Union[str, None, List[str]] = None, self_loop=True, agg="median") \
            -> Tuple[DataFrame, DataFrame]:
        """
        Combine relations for all `node_types` for a single LATTE layer.
        Args:
            node_types ():
            self_loop ():
            agg ():

        Returns:

        """
        if node_types is None:
            node_types = self._betas.keys()
        elif isinstance(node_types, str):
            node_types = [node_types]

        nid_offset = 0
        eid_offset = 0
        all_nodes = []
        all_links = []
        for ntype in node_types:
            if ntype not in self._betas: continue
            nodes, links = self.get_relation_attn(ntype, self_loop=self_loop, agg=agg)
            if nid_offset:
                nodes.index = nodes.index + nid_offset
                links['source'] = links['source'] + nid_offset
                links['target'] = links['target'] + nid_offset
                links.index = links.index + eid_offset

            all_nodes.append(nodes)
            all_links.append(links)

            nid_offset += nodes.index.size
            eid_offset += links.index.size

        if not len(all_nodes):
            return None, None

        all_nodes = pd.concat(all_nodes, axis=0)
        all_links = pd.concat(all_links, axis=0).sort_values(by=['mean', 'target'], ascending=False)

        # Set self-loops of dst nodes to src nodes
        for link_id, link in all_links.query('source == target').iterrows():
            ntype = all_nodes.loc[link['source'], 'label']
            src_node = all_nodes.query(f"label == '{ntype}'")['level'].idxmax()
            all_links.loc[link_id, 'source'] = src_node

        # Group duplicated src_nodes into one
        src_nodes = all_nodes.query("level == level.max()") \
            .sort_values('count', ascending=False) \
            .drop_duplicates(['label'])
        for nid, node in src_nodes.iterrows():
            prev_src_nodes = all_nodes.query(f'level == {node["level"]} and label == "{node["label"]}"').drop(
                index=[nid])
            replace_src_nodes = {prev_nid: nid for prev_nid in prev_src_nodes.index}
            all_nodes.drop(index=prev_src_nodes.index, inplace=True)
            all_links['source'].replace(replace_src_nodes, inplace=True)
            all_links['target'].replace(replace_src_nodes, inplace=True)

        all_nodes, all_links = reindex_contiguous(all_nodes, all_links)

        # assert all_links['source'].isin(all_nodes.index).all()
        # assert all_links['target'].isin(all_nodes.index).all()
        assert not all_nodes.index.duplicated().any(), all_nodes.index[all_nodes.index.duplicated()]
        assert not all_links.index.duplicated().any(), all_links.index[all_links.index.duplicated()]

        return all_nodes, all_links

    def get_relation_attn(self, ntype: str, self_loop=True, agg="median") \
            -> Tuple[DataFrame, DataFrame]:
        rel_attn = self._betas[ntype]

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
                                           "label": [metapath_name for i in range(len(targets))],
                                           "mean": [attn_agg for i in range(len(targets))],
                                           'std': [rel_attn_std.loc[metapath_name] for i in range(len(targets))]})
                links = links.append(path_links, ignore_index=True)


            elif self_loop:
                source = indexed_nodes[indexed_metapath[0]]

                links = links.append({"source": source, "target": source, "label": metapath_name,
                                      "mean": attn_agg, "std": rel_attn_std.loc[ntype], }, ignore_index=True)

        def _get_hash_label(metapaths):
            label = metapaths.str.split(".", expand=True)[1].fillna(metapaths).str.replace("rev_", "")
            return label

        links["color"] = _get_hash_label(links["label"]).apply(lambda label: ColorHash(label).hex)
        links = links.iloc[::-1]

        # Nodes
        # node_group = [int(node[0]) for node, nid in all_nodes.items()]
        # groups = [[nid for nid, node in enumerate(node_group) if node == group] for group in np.unique(node_group)]

        nodes = pd.DataFrame(columns=["label", "metapath", "level", "color", "count"])
        nodes["label"] = [node[1:] for node in indexed_nodes.keys()]
        nodes["level"] = [int(node[0]) for node in indexed_nodes.keys()]
        nodes["metapath"] = [indexed_node2metapath[node] for node in indexed_nodes.keys()]

        # Get number of edge_index counts for each metapath
        if ntype in self._counts:
            nodes["count"] = nodes["metapath"].map(lambda m: self._counts[ntype].sum(axis=0).get(m, 0)).astype(int)

        # Set count of target nodes
        nodes.loc[nodes.query(f'label == "{ntype}" & metapath == "{ntype}"').index, 'count'] = rel_attn.shape[0]

        nodes["color"] = nodes[["label", "level"]].apply(
            lambda x: ColorHash(x["label"].replace("rev_", ""), saturation=np.arange(0.2, 0.8, 0.1)).hex \
                if x["level"] % 2 == 0 \
                else ColorHash(x["label"], saturation=np.arange(0.2, 0.8, 0.2)).hex,
            axis=1)

        return nodes, links


def reindex_contiguous(layer_nodes: pd.DataFrame, layer_links: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    replace_nid = pd.Series(layer_nodes.reset_index().index, index=layer_nodes.index)
    replace_nid = {k: v for k, v in replace_nid.items() if k != v}
    layer_nodes = layer_nodes.reset_index(drop=True)
    layer_links = layer_links.replace({'source': replace_nid, 'target': replace_nid})
    return layer_nodes, layer_links


class RelationMultiLayerAgg:
    layers: List[RelationAttention]

    def get_metapaths_chain(self) -> List[List[Tuple[str, str, str]]]:
        metapaths_chain = {}
        for layer in self.layers:
            for metapath in layer.metapaths:
                if max_num_hops([metapath]) <= 1: continue
                chain = [metapath[i: i + 3] for i in range(0, len(metapath) - 1, 2)]
                metapaths_chain[metapath] = chain

        return list(metapaths_chain.values())

    @property
    def _beta_avg(self):
        return {i: layer._beta_avg for i, layer in enumerate(self.layers)}

    @property
    def _beta_std(self):
        return {i: layer._beta_std for i, layer in enumerate(self.layers)}

    def get_sankey_flow(self, node_types=None, self_loop=True, agg="median"):
        """
        Concatenate multiple layer's sankey flow.
        Args:
            node_types (): for compability, ignored.
            self_loop ():
            agg ():

        Returns:

        """
        nid_offset = 0
        eid_offset = 0
        layer_nodes = []
        layer_links = []
        last_src_nids = {}

        for latte in reversed(self.layers):
            layer_self_loop = self_loop if latte.layer + 1 <= len(self.layers) else False
            nodes, links = latte.get_sankey_flow(node_types=node_types, self_loop=layer_self_loop, agg=agg)
            nodes['layer'] = latte.layer
            links['layer'] = latte.layer

            if nid_offset:
                nodes.index = nodes.index + nid_offset
                links['source'] += nid_offset
                links['target'] += nid_offset
                links.index = links.index + eid_offset

            if len(layer_nodes) > 0:
                # Remove last layer's level=1 nodes, and replace current layer's level>1 node ids with last layer's level=1 nodes
                current_dst_ntypes = nodes.loc[nodes['level'] == 1]['label'].unique()

                for ntype, prev_src_id in last_src_nids.items():
                    if ntype not in current_dst_ntypes: continue

                    dst_nodes = nodes[(nodes['label'] == ntype) & (nodes['metapath'] == ntype) & nodes['level'] == 1]
                    dst_id = dst_nodes.index[0].item()
                    assert isinstance(dst_id, int) and dst_nodes.index.size == 1, dst_nodes

                    # Overwrite last layer's src node to current layer's dst node
                    nodes.drop(index=dst_id, inplace=True)

                    links['source'].replace(dst_id, prev_src_id, inplace=True)
                    links['target'].replace(dst_id, prev_src_id, inplace=True)

                if True:
                    # Ensure dst ntypes in non-last-layers have correct positioning by adding a self loop.
                    for ntype in set(current_dst_ntypes).difference(last_src_nids.keys()):
                        nid = nodes.query(f'(level == {nodes["level"].min()}) and (label == "{ntype}")').index[0]

                        selfloop_weight = 1 - links.query(f'target == {nid}')['mean'].sum() + 1e-3
                        selfloop_link = pd.Series({
                            'source': nid, 'target': nid, 'label': ntype, 'color': nodes.loc[nid, 'color'],
                            'mean': selfloop_weight, 'std': 0.0, 'layer': latte.layer},
                            name=links.index.max() + 1)
                        links = links.append(selfloop_link)

                nodes['level'] += layer_nodes[-1]['level'].max() - 1

            # Update last_src_nids to contain the src nodes from current layers
            last_src_nids = {}
            for id, node in nodes.loc[nodes['level'] == nodes['level'].max()].iterrows():
                last_src_nids[node['label']] = id  # Get the index value

            layer_nodes.append(nodes)
            layer_links.append(links)

            nid_offset += nodes.index.size
            eid_offset += links.index.size

        layer_nodes = pd.concat(layer_nodes, axis=0)
        layer_links = pd.concat(layer_links, axis=0).sort_values(by=['mean', 'target'], ascending=False)
        layer_nodes = layer_nodes.drop(columns=['metapath'])

        if len(layer_nodes) > 1:
            # Ensure node index is contiguous
            layer_nodes, layer_links = reindex_contiguous(layer_nodes, layer_links)

        if not layer_links['source'].isin(layer_nodes.index).all():
            print('layer_links[source] not in layer_nodes.index:',
                  pd.Index(layer_links['source']).difference(layer_nodes.index).values)
        if not layer_links['target'].isin(layer_nodes.index).all():
            print('layer_links[target] not in layer_nodes.index:',
                  pd.Index(layer_links['target']).difference(layer_nodes.index).values)
        assert not layer_nodes.index.duplicated().any(), \
            f"layer_nodes.index.duplicated(): \n{layer_nodes[layer_nodes.index.duplicated(keep=False)]}"
        assert not layer_links.index.duplicated().any(), \
            f"layer_links.index.duplicated(): \n{layer_links[layer_links.index.duplicated(keep=False)]}"

        return layer_nodes, layer_links
