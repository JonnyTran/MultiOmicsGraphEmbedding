import copy
import numpy as np
import pandas as pd
from typing import Union, Dict
from collections.abc import Iterable

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
    def __init__(self, t_order: int, embedding_dim: int, num_nodes_dict: dict, metapaths: list, batchnorm=False,
                 layernorm=False,
                 edge_dir="in", activation: str = "relu", attn_heads=1, attn_activation="sharpening", attn_dropout=0.5,
                 ):
        super(LATTE, self).__init__()
        self.t_order = t_order
        self.edge_dir = edge_dir
        self.node_types = list(num_nodes_dict.keys())
        self.metapaths = metapaths
        self.embedding_dim = embedding_dim * t_order

        layers = []
        for t in range(t_order):
            layers.append(
                LATTEConv(in_dim=embedding_dim,
                          embedding_dim=embedding_dim,
                          num_nodes_dict=num_nodes_dict,
                          metapaths=metapaths,
                          batchnorm=batchnorm,
                          layernorm=layernorm,
                          edge_dir=edge_dir,
                          activation=activation,
                          attn_heads=attn_heads,
                          attn_activation=attn_activation,
                          attn_dropout=attn_dropout,
                          first=True if t == 0 else False))
        self.layers = nn.ModuleList(layers)

    def forward(self, blocks: Union[Dict, DGLBlock], h_dict, **kwargs):

        for t in range(self.t_order):
            h_dict = self.layers[t].forward(blocks[t] if isinstance(blocks, Iterable) else blocks,
                                            h_dict, **kwargs)

        return h_dict


class LATTEConv(nn.Module):
    def __init__(self, in_dim, embedding_dim, num_nodes_dict: {str: int}, metapaths: list, batchnorm=False,
                 layernorm=False,
                 edge_dir="in", activation: str = "relu", attn_heads=4,
                 attn_activation="sharpening",
                 attn_dropout=0.2,
                 first=True) -> None:
        super(LATTEConv, self).__init__()
        self.first = first
        self.node_types = list(num_nodes_dict.keys())
        self.metapaths = list(metapaths)
        print(self.metapaths)
        self.edge_dir = edge_dir

        self.metapath_id = {etype: i for i, (srctype, etype, dsttype) in enumerate(self.metapaths)}
        self.num_nodes_dict = num_nodes_dict
        self.embedding_dim = embedding_dim
        self.attn_heads = attn_heads
        self.attn_dropout = attn_dropout

        if activation == "sigmoid":
            self.activation = F.sigmoid
        elif activation == "tanh":
            self.activation = F.tanh
        elif activation == "relu":
            self.activation = F.relu
        else:
            print(f"Embedding activation arg `{activation}` did not match, so uses linear activation.")

        self.linear_l = nn.ModuleDict(
            {node_type: nn.Linear(in_dim, embedding_dim, bias=True) \
             for node_type in self.node_types})  # W.shape (F x D_m)
        self.linear_r = nn.ModuleDict(
            {node_type: nn.Linear(in_dim, embedding_dim, bias=True) \
             for node_type in self.node_types})  # W.shape (F x D_m)

        self.out_channels = self.embedding_dim // attn_heads
        self.attn_l = nn.Parameter(torch.ones(len(self.metapaths), attn_heads, self.out_channels))
        self.attn_r = nn.Parameter(torch.ones(len(self.metapaths), attn_heads, self.out_channels))

        self.rel_attn_l = nn.ParameterDict({
            ntype: nn.Parameter(torch.ones(attn_heads, self.out_channels)) \
            for ntype in self.node_types})
        self.rel_attn_r = nn.ParameterDict({
            ntype: nn.Parameter(torch.ones(attn_heads, self.out_channels)) \
            for ntype in self.node_types})

        if layernorm:
            self.layernorm = nn.ModuleDict({
                ntype: nn.LayerNorm(embedding_dim) for ntype in self.node_types})

        if batchnorm:
            self.batchnorm = torch.nn.ModuleDict({
                ntype: torch.nn.BatchNorm1d(embedding_dim) for ntype in self.node_types})

        if attn_activation == "sharpening":
            self.alpha_activation = nn.Parameter(torch.ones(len(self.metapaths)))
        elif attn_activation == "PReLU":
            self.alpha_activation = nn.PReLU(init=0.2)
        elif attn_activation == "LeakyReLU":
            self.alpha_activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            print(f"WARNING: alpha_activation `{attn_activation}` did not match, so used linear activation")
            self.alpha_activation = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)

        for ntype in self.node_types:
            nn.init.xavier_uniform_(self.rel_attn_l[ntype])
            nn.init.xavier_uniform_(self.rel_attn_r[ntype])

            nn.init.xavier_uniform_(self.linear_l[ntype].weight)
            nn.init.xavier_uniform_(self.linear_r[ntype].weight)

    def edge_attention(self, edges: EdgeBatch):
        srctype, etype, dsttype = edges.canonical_etype

        att_l = (edges.src["k"] * self.attn_l[self.metapath_id[etype]]).sum(dim=-1)
        att_r = (edges.dst["v"] * self.attn_r[self.metapath_id[etype]]).sum(dim=-1)

        att = att_l + att_r
        att = F.dropout(att, p=self.attn_dropout, training=self.training)
        att = self.alpha_activation(att) if isinstance(self.alpha_activation, nn.Module) else \
            att * self.alpha_activation[self.metapath_id[etype]]

        return {etype: att, "h": edges.dst["v"]}

    def message_func(self, edges: EdgeBatch):
        srctype, etype, dsttype = edges.canonical_etype

        return {etype: edges.data[etype], "h": edges.data["h"]}

    def reduce_func(self, nodes: NodeBatch):
        '''
            Softmax based on target node's id (edge_index_i).
        '''
        output = {}
        for srctype, etype, dsttype in self.metapaths:
            if etype not in nodes.mailbox: continue

            att = F.softmax(nodes.mailbox[etype], dim=1)
            h = torch.sum(att.unsqueeze(dim=-1) * nodes.mailbox['h'], dim=1)
            output[etype] = h

        return output

    def get_beta_weights(self, node_emb, rel_embs, ntype):
        alpha_l = (node_emb * self.rel_attn_l[ntype]).sum(dim=-1)
        alpha_r = (rel_embs * self.rel_attn_r[ntype][None, :, :]).sum(dim=-1)

        beta = alpha_l[:, None, :] + alpha_r
        beta = F.leaky_relu(beta, negative_slope=0.2)
        beta = F.softmax(beta, dim=2)
        beta = F.dropout(beta, p=self.attn_dropout, training=self.training)
        return beta

    def forward(self, g: Union[DGLBlock, DGLHeteroGraph], feat: dict):
        feat_src, feat_dst = expand_as_pair(input_=feat, g=g)

        funcs = {}
        for srctype in set(srctype for srctype, etype, dsttype in g.canonical_etypes):
            g.srcnodes[srctype].data['k'] = self.linear_l[srctype](feat_src[srctype]) \
                .view(-1, self.attn_heads, self.out_channels)

        for dsttype in set(dsttype for srctype, etype, dsttype in g.canonical_etypes):
            g.dstnodes[dsttype].data['v'] = self.linear_r[dsttype](feat_dst[dsttype]) \
                .view(-1, self.attn_heads, self.out_channels)

        for srctype, etype, dsttype in g.canonical_etypes:
            # Compute node-level attention coefficients
            g.apply_edges(func=self.edge_attention, etype=etype)

            if g.batch_num_edges(etype=etype).nelement() > 1 or g.batch_num_edges(etype=etype).item() > 0:
                funcs[etype] = (self.message_func, self.reduce_func)

        g.multi_update_all(funcs, cross_reducer='mean')

        # For each metapath in a node_type, use GAT message passing to aggregate h_j neighbors
        out = {}
        for ntype in set(g.ntypes):
            etypes = [etype for etype in self.get_head_relations(ntype, etype_only=True) \
                      if etype in g.dstnodes[ntype].data]

            # If node type doesn't have any messages
            if len(etypes) == 0:
                out[ntype] = feat_dst[ntype]
                continue
            # If homogeneous graph
            # if len(g.etypes) == 1:
            #     out[ntype] = out[ntype] = torch.stack(
            #         [g.dstnodes[ntype].data[etype] for etype in etypes] + [
            #             g.dstnodes[ntype].data["feat"].view(-1, self.embedding_dim), ], dim=1)
            #     if hasattr(self, "activation"):
            #         out[ntype] = self.activation(out[ntype])
            #     continue

            # Soft-select the relation-specific embeddings by a weighted average with beta[node_type]
            out[ntype] = torch.stack(
                [g.dstnodes[ntype].data[etype] for etype in etypes] + [
                    g.dstnodes[ntype].data["feat"].view(-1, self.attn_heads, self.out_channels), ], dim=1)
            # out[ntype] = torch.mean(out[ntype], dim=1)

            beta = self.get_beta_weights(node_emb=out[ntype][:, -1, :],
                                         rel_embs=out[ntype],
                                         ntype=ntype)
            out[ntype] = out[ntype] * beta.unsqueeze(-1)
            out[ntype] = out[ntype].sum(1).view(out[ntype].size(0), self.embedding_dim)

            if hasattr(self, "layernorm"):
                out[ntype] = self.layernorm[ntype](out[ntype])

            if hasattr(self, "batchnorm"):
                out[ntype] = self.batchnorm[ntype](out[ntype])

            if hasattr(self, "activation"):
                out[ntype] = self.activation(out[ntype])

        return out

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


