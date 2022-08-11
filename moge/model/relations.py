from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
from colorhash import ColorHash
from pandas import DataFrame
from torch import Tensor

from moge.model.PyG.utils import filter_metapaths


class RelationAttention:
    metapaths: List[Tuple[str, str, str]]

    def __init__(self):
        self._betas = {}
        self._beta_std = {}
        self._beta_avg = {}

    def get_head_relations(self, head_node_type, order=None, str_form=False) -> List[Tuple[str, str, str]]:
        relations = filter_metapaths(self.metapaths, order=order, head_type=head_node_type)

        if str_form:
            relations = [".".join(metapath) if isinstance(metapath, tuple) else metapath \
                         for metapath in relations]

        return relations

    def get_tail_relations(self, tail_node_type, order=None, str_form=False) -> List[Tuple[str, str, str]]:
        relations = filter_metapaths(self.metapaths, order=order, tail_type=tail_node_type)

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

    def save_relation_weights(self, betas: Dict[str, Tensor], global_node_idx: Dict[str, Tensor]):
        # Only save relation weights if beta has weights for all node_types in the global_node_idx batch
        if len(betas) < len({metapath[-1] for metapath in self.metapaths}):
            return

        for ntype in betas:
            if ntype not in global_node_idx or global_node_idx[ntype].numel() == 0: continue

            relations = self.get_tail_relations(ntype, str_form=True) + [ntype, ]
            if len(relations) <= 1: continue

            with torch.no_grad():
                df = pd.DataFrame(betas[ntype].squeeze(-1).cpu().numpy(),
                                  columns=relations,
                                  index=global_node_idx[ntype].cpu().numpy(), dtype=np.float16)

                if len(self._betas) == 0 or ntype not in self._betas:
                    self._betas[ntype] = df
                else:
                    self._betas[ntype].update(df, overwrite=True)

                # Compute mean and std of beta scores across all nodes
                _beta_avg = np.around(betas[ntype].mean(dim=0).squeeze(-1).cpu().numpy(), decimals=3)
                _beta_std = np.around(betas[ntype].std(dim=0).squeeze(-1).cpu().numpy(), decimals=2)
                self._beta_avg[ntype] = {metapath: _beta_avg[i] for i, metapath in enumerate(relations)}
                self._beta_std[ntype] = {metapath: _beta_std[i] for i, metapath in enumerate(relations)}

    def get_relation_weights(self, std=True):
        """
        Get the mean and std of relation attention weights for all nodes
        :return:
        """
        _beta_avg = {}
        _beta_std = {}
        for ntype in self._betas:
            relations = self.get_tail_relations(ntype, str_form=True) + [ntype, ]
            _beta_avg = np.around(self._betas[ntype].mean(), decimals=3)
            _beta_std = np.around(self._betas[ntype].std(), decimals=2)
            self._beta_avg[ntype] = {metapath: _beta_avg[i] for i, metapath in
                                     enumerate(relations)}
            self._beta_std[ntype] = {metapath: _beta_std[i] for i, metapath in
                                     enumerate(relations)}

        print_output = {}
        for node_type in self._beta_avg:
            for metapath, avg in self._beta_avg[node_type].items():
                if std:
                    print_output[metapath] = (avg, self._beta_std[node_type][metapath])
                else:
                    print_output[metapath] = avg
        return print_output

    def get_sankey_flow(self, node_type: str, self_loop: bool = False, agg="mean") \
            -> Tuple[DataFrame, DataFrame]:

        rel_attn: pd.DataFrame = self._betas[node_type]

        if agg == "sum":
            rel_attn = rel_attn.sum(axis=0)
        elif agg == "median":
            rel_attn = rel_attn.median(axis=0)
        elif agg == "max":
            rel_attn = rel_attn.max(axis=0)
        elif agg == "min":
            rel_attn = rel_attn.min(axis=0)
        else:
            rel_attn = rel_attn.mean(axis=0)

        indexed_metapaths = rel_attn.index.str.split(".").map(lambda tup: [str(len(tup) - i) + name \
                                                                           for i, name in enumerate(tup)])
        all_nodes = {node for nodes in indexed_metapaths for node in nodes}
        all_nodes = {node: i for i, node in enumerate(all_nodes)}

        # Links
        links = pd.DataFrame(columns=["source", "target", "value", "label", "color"])
        for i, (metapath, value) in enumerate(rel_attn.to_dict().items()):
            indexed_metapath = indexed_metapaths[i]

            if len(metapath.split(".")) > 1:
                sources = [all_nodes[indexed_metapath[j]] for j, _ in enumerate(indexed_metapath[:-1])]
                targets = [all_nodes[indexed_metapath[j + 1]] for j, _ in enumerate(indexed_metapath[:-1])]

                path_links = pd.DataFrame({"source": sources, "target": targets,
                                           "value": [value, ] * len(targets), "label": [metapath, ] * len(targets)})
                links = links.append(path_links, ignore_index=True)


            elif self_loop:
                source = all_nodes[indexed_metapath[0]]
                links = links.append({"source": source, "target": source,
                                      "value": value, "label": metapath}, ignore_index=True)

        links["color"] = links["label"].apply(lambda label: ColorHash(label).hex)
        links = links.iloc[::-1]

        # Nodes
        # node_group = [int(node[0]) for node, nid in all_nodes.items()]
        # groups = [[nid for nid, node in enumerate(node_group) if node == group] for group in np.unique(node_group)]

        nodes = pd.DataFrame(columns=["label", "level", "color"])
        nodes["label"] = [node[1:] for node in all_nodes.keys()]
        nodes["level"] = [int(node[0]) for node in all_nodes.keys()]

        nodes["color"] = nodes[["label", "level"]].apply(
            lambda x: ColorHash(x["label"].strip("rev_")).hex \
                if x["level"] % 2 == 0 \
                else ColorHash(x["label"]).hex, axis=1)

        return nodes, links
