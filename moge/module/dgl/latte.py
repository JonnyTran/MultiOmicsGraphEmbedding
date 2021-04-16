import copy
import numpy as np
import pandas as pd

import torch
from torch import nn as nn

import torch.nn.functional as F

import pytorch_lightning as pl

import dgl
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch import edge_softmax, GATConv
from dgl.utils import expand_as_pair
from dgl.udf import EdgeBatch, NodeBatch
from dgl.heterograph import DGLHeteroGraph, DGLBlock

from moge.module.sampling import negative_sample, negative_sample_head_tail
from moge.module.utils import preprocess_input, tensor_sizes


class LATTE(nn.Module):
    def __init__(self, t_order: int, embedding_dim: int, in_channels_dict: dict, num_nodes_dict: dict, metapaths: list,
                 edge_dir="in", activation: str = "relu", attn_heads=1, attn_activation="sharpening", attn_dropout=0.5,
                 use_proximity=True, neg_sampling_ratio=2.0):
        super(LATTE, self).__init__()
        self.metapaths = metapaths
        self.edge_dir = edge_dir
        self.node_types = list(num_nodes_dict.keys())
        self.embedding_dim = embedding_dim * t_order
        self.use_proximity = use_proximity
        self.t_order = t_order
        self.neg_sampling_ratio = neg_sampling_ratio

        layers = []
        t_order_metapaths = copy.deepcopy(metapaths)
        for t in range(t_order):
            layers.append(
                LATTEConv(embedding_dim=embedding_dim,
                          in_channels_dict=in_channels_dict,
                          num_nodes_dict=num_nodes_dict,
                          metapaths=metapaths, edge_dir=edge_dir, activation=activation, attn_heads=attn_heads,
                          attn_activation=attn_activation, attn_dropout=attn_dropout, use_proximity=use_proximity,
                          neg_sampling_ratio=neg_sampling_ratio, first=True if t == 0 else False, cpu_embeddings=None))
            t_order_metapaths = LATTE.join_metapaths(t_order_metapaths, metapaths)
        self.layers = nn.ModuleList(layers)

    @staticmethod
    def join_metapaths(metapath_A, metapath_B):
        metapaths = []
        for relation_a in metapath_A:
            for relation_b in metapath_B:
                if relation_a[-1] == relation_b[0]:
                    new_relation = relation_a + relation_b[1:]
                    metapaths.append(new_relation)
        return metapaths

    @staticmethod
    def get_edge_index_values(edge_index_tup: [tuple, torch.Tensor]):
        if isinstance(edge_index_tup, tuple):
            edge_index = edge_index_tup[0]
            edge_values = edge_index[1]

            if edge_values.dtype != torch.float:
                edge_values = edge_values.to(torch.float)
        elif isinstance(edge_index_tup, torch.Tensor) and edge_index_tup.size(1) > 0:
            edge_index = edge_index_tup
            edge_values = torch.ones_like(edge_index_tup[0], dtype=torch.float)
        else:
            return None, None

        return edge_index, edge_values

    @staticmethod
    def join_edge_indexes(edge_index_dict_A, edge_index_dict_B, global_node_idx):
        output_dict = {}
        for metapath_a, edge_index_a in edge_index_dict_A.items():
            if is_negative(metapath_a): continue
            edge_index_a, values_a = LATTE.get_edge_index_values(edge_index_a)
            if edge_index_a is None: continue

            for metapath_b, edge_index_b in edge_index_dict_B.items():
                if metapath_a[-1] != metapath_b[0] or is_negative(metapath_b): continue

                new_metapath = metapath_a + metapath_b[1:]
                edge_index_b, values_b = LATTE.get_edge_index_values(edge_index_b)
                if edge_index_b is None: continue
                try:
                    new_edge_index = adamic_adar(indexA=edge_index_a, valueA=values_a, indexB=edge_index_b,
                                                 valueB=values_b,
                                                 m=global_node_idx[metapath_a[0]].size(0),
                                                 k=global_node_idx[metapath_a[-1]].size(0),
                                                 n=global_node_idx[metapath_b[-1]].size(0),
                                                 coalesced=True, sampling=True)
                    if new_edge_index[0].size(1) == 0: continue
                    output_dict[new_metapath] = new_edge_index

                except Exception as e:
                    print(f"{str(e)} \n {metapath_a}: {edge_index_a.size(1)}, {metapath_b}: {edge_index_b.size(1)}")
                    continue

        return output_dict

    def forward(self, blocks, feat, **kwargs):
        """
        This
        :param X: Dict of <node_type>:<tensor size (batch_size, in_channels)>. If nodes are not attributed, then pass an empty dict.
        :param global_node_idx: Dict of <node_type>:<int tensor size (batch_size,)>
        :param edge_index_dict: Dict of <metapath>:<tensor size (2, num_edge_index)>
        :param save_betas: whether to save _beta values for batch
        :return embedding_output, proximity_loss, edge_pred_dict:
        """
        # proximity_loss = torch.tensor(0.0, device=self.layers[0].device) if self.use_proximity else None

        # h_layers = {node_type: [] for node_type in self.node_types}
        for t in range(self.t_order):
            if t == 0:
                h_dict, t_loss, edge_pred_dict = self.layers[t].forward(blocks[t], feat, **kwargs)
            else:
                h_dict, t_loss, _ = self.layers[t].forward(blocks[t], h_dict, **kwargs)

            # for node_type in h_dict:
            #     h_layers[node_type].append(h_dict[node_type])

            # if self.use_proximity:
            #     proximity_loss += t_loss

        # concat_out = {node_type: torch.cat(h_list, dim=1) for node_type, h_list in h_layers.items() \
        #               if len(h_list) > 0 and h_list[0].size(0) != 0}

        return h_dict

    def get_attn_activation_weights(self, t):
        return dict(zip(self.layers[t].metapaths, self.layers[t].alpha_activation.detach().numpy().tolist()))

    def get_relation_weights(self, t):
        return self.layers[t].get_relation_weights()


class LATTEConv(nn.Module):
    def __init__(self, embedding_dim: int, in_channels_dict: {str: int}, num_nodes_dict: {str: int}, metapaths: list,
                 edge_dir="in", activation: str = "relu", attn_heads=4, attn_activation="sharpening", attn_dropout=0.2,
                 use_proximity=False, neg_sampling_ratio=1.0, first=True, cpu_embeddings=None) -> None:
        super(LATTEConv, self).__init__()
        self.first = first
        self.node_types = list(num_nodes_dict.keys())
        self.metapaths = list(metapaths)
        print(self.metapaths)
        self.edge_dir = edge_dir

        self.metapath_id = {etype: i for i, (srctype, etype, dsttype) in enumerate(self.metapaths)}
        self.num_nodes_dict = num_nodes_dict
        self.embedding_dim = embedding_dim
        self.use_proximity = use_proximity
        self.neg_sampling_ratio = neg_sampling_ratio
        self.attn_heads = attn_heads
        self.attn_dropout = attn_dropout

        self.activation = activation.lower()
        if self.activation not in ["sigmoid", "tanh", "relu"]:
            print(f"Embedding activation arg `{self.activation}` did not match, so uses linear activation.")

        self.conv = torch.nn.ModuleDict(
            {node_type: torch.nn.Conv1d(
                in_channels=in_channels_dict[
                    node_type] if first and node_type in in_channels_dict else embedding_dim,
                out_channels=self.num_head_relations(node_type),
                kernel_size=1) \
                for node_type in self.node_types})  # W_phi.shape (D x F)

        if first:
            self.linear_l = nn.ModuleDict(
                {node_type: nn.Linear(in_channels, embedding_dim, bias=True) \
                 for node_type, in_channels in in_channels_dict.items()})  # W.shape (F x D_m)
            self.linear_r = nn.ModuleDict(
                {node_type: nn.Linear(in_channels, embedding_dim, bias=True) \
                 for node_type, in_channels in in_channels_dict.items()})  # W.shape (F x D_m)
        else:
            self.linear_l = nn.ModuleDict(
                {node_type: nn.Linear(embedding_dim, embedding_dim, bias=True) \
                 for node_type in self.node_types})  # W.shape (F x F)
            self.linear_r = nn.ModuleDict(
                {node_type: nn.Linear(embedding_dim, embedding_dim, bias=True) \
                 for node_type in self.node_types})  # W.shape (F x F}

        self.out_channels = self.embedding_dim // attn_heads
        self.attn_l = nn.Parameter(torch.Tensor(len(self.metapaths), attn_heads, self.out_channels))
        self.attn_r = nn.Parameter(torch.Tensor(len(self.metapaths), attn_heads, self.out_channels))

        if attn_activation == "sharpening":
            self.alpha_activation = nn.Parameter(torch.Tensor(len(self.metapaths)).fill_(1.0))
        elif attn_activation == "PReLU":
            self.alpha_activation = nn.PReLU(init=0.2)
        elif attn_activation == "LeakyReLU":
            self.alpha_activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            print(f"WARNING: alpha_activation `{attn_activation}` did not match, so used linear activation")
            self.alpha_activation = None

        # If some node type are not attributed, instantiate embeddings for them
        non_attr_node_types = (num_nodes_dict.keys() - in_channels_dict.keys())
        if first and len(non_attr_node_types) > 0:
            print(f"embeddings needed for {non_attr_node_types}")

    def reset_parameters(self):
        for i, metapath in enumerate(self.metapaths):
            nn.init.xavier_uniform_(self.attn_l[i])
            nn.init.xavier_uniform_(self.attn_r[i])

        for node_type in self.linear_l:
            nn.init.xavier_uniform_(self.linear[node_type].weight)
        for node_type in self.conv:
            nn.init.xavier_uniform_(self.conv[node_type].weight)

        if self.embeddings is not None and len(self.embeddings.keys()) > 0:
            for node_type in self.embeddings:
                self.embeddings[node_type].reset_parameters()

    def edge_attention(self, edges: EdgeBatch):
        srctype, etype, dsttype = edges.canonical_etype
        att_l = (edges.src["k"] * self.attn_l[self.metapath_id[etype]]).sum(dim=-1)
        att_r = (edges.dst["v"] * self.attn_r[self.metapath_id[etype]]).sum(dim=-1)
        att = att_l + att_r

        return {etype: att, "h": edges.dst["v"]}

    def message_func(self, edges: EdgeBatch):
        srctype, etype, dsttype = edges.canonical_etype

        return {etype: edges.data[etype],
                "h": edges.data["h"]}

    def reduce_func(self, nodes: NodeBatch):
        '''
            Softmax based on target node's id (edge_index_i).
        '''
        output = {}
        for srctype, etype, dsttype in self.metapaths:
            if etype not in nodes.mailbox: continue

            att = F.softmax(nodes.mailbox[etype], dim=1)
            h = torch.sum(att.unsqueeze(dim=-1) * nodes.mailbox['h'], dim=1)
            output[etype] = h.view(-1, self.embedding_dim)

        return output

    def forward(self, g: DGLBlock, feat: dict):
        feat_src, feat_dst = expand_as_pair(input_=feat, g=g)

        with g.local_scope():
            funcs = {}
            for srctype in set(srctype for srctype, etype, dsttype in g.canonical_etypes):
                g.srcnodes[srctype].data['k'] = self.linear_l[srctype](feat_src[srctype]).view(-1, self.attn_heads,
                                                                                               self.out_channels)
                # G.srcnodes[srctype].data['v'] = v_linear(feat_src[srctype]).view(-1, self.n_heads, self.d_k)

            for dsttype in set(dsttype for srctype, etype, dsttype in g.canonical_etypes):
                g.dstnodes[dsttype].data['v'] = self.linear_r[dsttype](feat_dst[dsttype]).view(-1, self.attn_heads,
                                                                                               self.out_channels)

            for srctype, etype, dsttype in g.canonical_etypes:
                # Compute node-level attention coefficients
                g.apply_edges(func=self.edge_attention, etype=etype)

                if g.batch_num_edges(etype=etype).item() > 0:
                    funcs[etype] = (self.message_func, self.reduce_func)

            g.multi_update_all(funcs, cross_reducer='mean')

            # For each metapath in a node_type, use GAT message passing to aggregate h_j neighbors
            out = {}
            for ntype in set(g.ntypes):
                etypes = [etype for etype in self.get_head_relations(ntype, etype_only=True) \
                          if etype in g.dstnodes[ntype].data]

                if len(etypes) == 0:
                    out[ntype] = feat_dst[ntype][:, :self.embedding_dim]
                    continue

                # Soft-select the relation-specific embeddings by a weighted average with beta[node_type]
                out[ntype] = torch.stack(
                    [g.dstnodes[ntype].data[etype] for etype in etypes] + [
                        g.dstnodes[ntype].data["v"].view(-1, self.embedding_dim), ], dim=1)
                out[ntype] = torch.mean(out[ntype], dim=1)
                # out[ntype] = torch.bmm(out[ntype].permute(0, 2, 1), beta[ntype]).squeeze(-1)

                # Apply \sigma activation to all embeddings
                out[ntype] = self.embedding_activation(out[ntype])

        proximity_loss, edge_pred_dict = None, None
        # if self.use_proximity:
        #     proximity_loss, edge_pred_dict = self.proximity_loss(edge_index_dict,
        #                                                          alpha_l=alpha_l, alpha_r=alpha_r,
        #                                                          global_node_idx=global_node_idx)
        return out, proximity_loss, edge_pred_dict


    def predict_scores(self, edge_index, alpha_l, alpha_r, metapath, logits=False):
        assert metapath in self.metapaths, f"If metapath `{metapath}` is tag_negative()'ed, then pass it with untag_negative()"

        # e_pred = self.attn_q[self.metapaths.index(metapath)].forward(
        #     torch.cat([alpha_l[metapath][edge_index[0]], alpha_r[metapath][edge_index[1]]], dim=1)).squeeze(-1)

        e_pred = self.attn_activation(alpha_l[metapath][edge_index[0]] + alpha_r[metapath][edge_index[1]],
                                      metapath_id=self.metapaths.index(metapath)).squeeze(-1)
        if logits:
            return e_pred
        else:
            return F.sigmoid(e_pred)

    def proximity_loss(self, edge_index_dict, alpha_l, alpha_r, global_node_idx):
        """
        For each relation/metapath type given in `edge_index_dict`, this function both predict link scores and computes
        the NCE loss for both positive and negative (sampled) links. For each relation type in `edge_index_dict`, if the
        negative metapath is not included, then the function automatically samples for random negative edges. And, if it
        is included, then computes the NCE loss over the given negative edges. This function returns the scores of the
        predicted positive and negative edges.

        :param edge_index_dict (dict): Dict of <relation/metapath>: <Tensor(2, num_edges)>
        :param alpha_l (dict): Dict of <node_type>:<alpha_l tensor>
        :param alpha_r (dict): Dict of <node_type>:<alpha_r tensor>
        :param global_node_idx (dict): Dict of <node_type>:<Tensor(node_idx,)>
        :return loss, edge_pred_dict: NCE loss. edge_pred_dict will contain both positive relations of shape (num_edges,) and negative relations of shape (num_edges*num_neg_edges, )
        """
        loss = torch.tensor(0.0, dtype=torch.float, device=self.conv[self.node_types[0]].weight.device)
        edge_pred_dict = {}
        for metapath, edge_index in edge_index_dict.items():
            # KL Divergence over observed positive edges or negative edges (if included)
            if isinstance(edge_index, tuple):  # Weighted edges
                edge_index, values = edge_index
            else:
                values = 1.0
            if edge_index is None: continue

            if not is_negative(metapath):
                e_pred_logits = self.predict_scores(edge_index, alpha_l, alpha_r, metapath, logits=True)
                loss += -torch.mean(values * F.logsigmoid(e_pred_logits), dim=-1)
            elif is_negative(metapath):
                e_pred_logits = self.predict_scores(edge_index, alpha_l, alpha_r, untag_negative(metapath), logits=True)
                loss += -torch.mean(F.logsigmoid(-e_pred_logits), dim=-1)

            edge_pred_dict[metapath] = F.sigmoid(e_pred_logits.detach())

            # Only need to sample for negative edges if negative metapath is not included
            if not is_negative(metapath) and tag_negative(metapath) not in edge_index_dict:
                neg_edge_index = negative_sample(edge_index,
                                                 M=global_node_idx[metapath[0]].size(0),
                                                 N=global_node_idx[metapath[-1]].size(0),
                                                 n_sample_per_edge=self.neg_sampling_ratio)
                if neg_edge_index is None or neg_edge_index.size(1) <= 1: continue

                e_neg_logits = self.predict_scores(neg_edge_index, alpha_l, alpha_r, metapath, logits=True)
                loss += -torch.mean(F.logsigmoid(-e_neg_logits), dim=-1)
                edge_pred_dict[tag_negative(metapath)] = F.sigmoid(e_neg_logits.detach())

        loss = torch.true_divide(loss, max(len(edge_index_dict) * 2, 1))
        return loss, edge_pred_dict

    def embedding_activation(self, embeddings):
        if self.activation == "sigmoid":
            return F.sigmoid(embeddings)
        elif self.activation == "tanh":
            return F.tanh(embeddings)
        elif self.activation == "relu":
            return F.relu(embeddings)
        else:
            return embeddings

    def attn_activation(self, alpha, metapath_id):
        if isinstance(self.alpha_activation, torch.Tensor):
            return self.alpha_activation[metapath_id] * alpha
        elif isinstance(self.alpha_activation, nn.Module):
            return self.alpha_activation.forward(alpha)
        else:
            return alpha

    def get_head_relations(self, head_node_type, to_str=False, etype_only=False) -> list:
        if self.edge_dir == "out":
            relations = [metapath \
                         for metapath in self.metapaths \
                         if metapath[0] == head_node_type]
        elif self.edge_dir == "in":
            relations = [metapath \
                         for metapath in self.metapaths \
                         if metapath[-1] == head_node_type]

        if to_str:
            relations = [".".join(metapath) if isinstance(metapath, tuple) else metapath \
                         for metapath in relations]
        if etype_only:
            relations = [metapath[1] if isinstance(metapath, tuple) else metapath \
                         for metapath in relations]

        return relations

    def num_head_relations(self, node_type) -> int:
        """
        Return the number of metapaths with head node type equals to :param node_type: and plus one for none-selection.
        :param node_type (str):
        :return:
        """
        relations = self.get_head_relations(node_type)
        return len(relations) + 1

    def save_relation_weights(self, beta, global_node_idx):
        # Only save relation weights if beta has weights for all node_types in the global_node_idx batch
        if len(beta) < len(self.node_types): return

        self._betas = {}
        self._beta_avg = {}
        self._beta_std = {}
        for node_type in beta:
            with torch.no_grad():
                self._betas[node_type] = pd.DataFrame(beta[node_type].squeeze(-1).cpu().numpy(),
                                                      columns=self.get_head_relations(node_type, True) + [node_type, ],
                                                      index=global_node_idx[node_type].cpu().numpy())

                _beta_avg = np.around(beta[node_type].mean(dim=0).squeeze(-1).cpu().numpy(), decimals=3)
                _beta_std = np.around(beta[node_type].std(dim=0).squeeze(-1).cpu().numpy(), decimals=2)
                self._beta_avg[node_type] = {metapath: _beta_avg[i] for i, metapath in
                                             enumerate(self.get_head_relations(node_type, True) + [node_type])}
                self._beta_std[node_type] = {metapath: _beta_std[i] for i, metapath in
                                             enumerate(self.get_head_relations(node_type, True) + [node_type])}

    def save_attn_weights(self, node_type, attn_weights, node_idx):
        if not hasattr(self, "_betas"):
            self._betas = {}
        if not hasattr(self, "_beta_avg"):
            self._beta_avg = {}
        if not hasattr(self, "_beta_std"):
            self._beta_std = {}

        betas = attn_weights.sum(1)
        with torch.no_grad():
            self._betas[node_type] = pd.DataFrame(betas.cpu().numpy(),
                                                  columns=self.get_head_relations(node_type, True) + [node_type, ],
                                                  index=node_idx.cpu().numpy())

            _beta_avg = np.around(betas.mean(dim=0).cpu().numpy(), decimals=3)
            _beta_std = np.around(betas.std(dim=0).cpu().numpy(), decimals=2)
            self._beta_avg[node_type] = {metapath: _beta_avg[i] for i, metapath in
                                         enumerate(self.get_head_relations(node_type, True) + [node_type])}
            self._beta_std[node_type] = {metapath: _beta_std[i] for i, metapath in
                                         enumerate(self.get_head_relations(node_type, True) + [node_type])}

    def get_relation_weights(self):
        """
        Get the mean and std of relation attention weights for all nodes
        :return:
        """
        return {(metapath if "." in metapath or len(metapath) > 1 else node_type): (avg, std) \
                for node_type in self._beta_avg for (metapath, avg), (relation_b, std) in
                zip(self._beta_avg[node_type].items(), self._beta_std[node_type].items())}


def tag_negative(metapath):
    if isinstance(metapath, tuple):
        return metapath + ("neg",)
    elif isinstance(metapath, str):
        return metapath + "_neg"
    else:
        return "neg"


def untag_negative(metapath):
    if isinstance(metapath, tuple) and metapath[-1] == "neg":
        return metapath[:-1]
    elif isinstance(metapath, str):
        return metapath.strip("_neg")
    else:
        return metapath


def is_negative(metapath):
    if isinstance(metapath, tuple) and metapath[-1] == "neg":
        return True
    elif isinstance(metapath, str) and "_neg" in metapath:
        return True
    else:
        return False


def adamic_adar(indexA, valueA, indexB, valueB, m, k, n, coalesced=False, sampling=True):
    A = SparseTensor(row=indexA[0], col=indexA[1], value=valueA,
                     sparse_sizes=(m, k), is_sorted=not coalesced)
    B = SparseTensor(row=indexB[0], col=indexB[1], value=valueB,
                     sparse_sizes=(k, n), is_sorted=not coalesced)

    deg_A = A.storage.colcount()
    deg_B = B.storage.rowcount()
    deg_normalized = 1.0 / (deg_A + deg_B).to(torch.float)
    deg_normalized[deg_normalized == float('inf')] = 0.0

    D = SparseTensor(row=torch.arange(deg_normalized.size(0), device=valueA.device),
                     col=torch.arange(deg_normalized.size(0), device=valueA.device),
                     value=deg_normalized.type_as(valueA),
                     sparse_sizes=(deg_normalized.size(0), deg_normalized.size(0)))

    out = A @ D @ B
    row, col, values = out.coo()

    num_samples = min(int(valueA.numel()), int(valueB.numel()), values.numel())
    if sampling and values.numel() > num_samples:
        idx = torch.multinomial(values, num_samples=num_samples,
                                replacement=False)
        row, col, values = row[idx], col[idx], values[idx]

    return torch.stack([row, col], dim=0), values
