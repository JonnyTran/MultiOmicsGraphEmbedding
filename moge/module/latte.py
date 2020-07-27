import copy, random
import numpy as np
import torch
from torch import nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import pytorch_lightning as pl


class LATTE(nn.Module):
    def __init__(self, t_order: int, embedding_dim: int, num_nodes_dict: dict, node_attr_shape: dict, metapaths: list,
                 use_proximity_loss=True):
        super(LATTE, self).__init__()
        self.metapaths = metapaths
        self.use_proximity_loss = use_proximity_loss
        self.t_order = t_order

        layers = []
        t_order_metapaths = copy.copy(metapaths)
        for t in range(1, t_order + 1):
            layers.append(LATTELayer(t_order=t, embedding_dim=embedding_dim, num_nodes_dict=num_nodes_dict,
                                     node_attr_shape=node_attr_shape, metapaths=t_order_metapaths))
            t_order_metapaths = self.join_relations(t_order_metapaths, metapaths)

        self.layers = nn.ModuleList(layers)

    def join_relations(self, metapath_A, metapath_B):
        metapaths = []
        for relation_a in metapath_A:
            for relation_b in metapath_B:
                if relation_a[-1] == relation_b[0]:
                    new_relation = relation_a + relation_b[1:]
                    metapaths.append(new_relation)
        return metapaths

    def forward(self, x_dict, x_index_dict, edge_index_dict):
        t_order_metapaths = edge_index_dict
        proximity_loss = 0
        for t in range(1, self.t_order + 1):
            h_dict, t_order_loss = self.layers[t - 1].forward(x_dict, x_index_dict, edge_index_dict)
            t_order_metapaths = self.join_relations(t_order_metapaths, edge_index_dict.keys())

            if self.use_proximity_loss:
                proximity_loss += t_order_loss

        embedding_output = None

        return embedding_output, proximity_loss

    def loss(self):
        pass


class LATTELayer(MessagePassing, pl.LightningModule):
    def __init__(self, t_order: int, embedding_dim: int, num_nodes_dict: {str: int}, node_attr_shape: {str: int},
                 metapaths: list, use_proximity_loss=True, neg_sampling_ratio=1.0) -> None:
        super(LATTELayer, self).__init__(aggr="add", flow="target_to_source", node_dim=0)
        assert t_order > 0, "t_order must start from 1"
        self.t_order = t_order
        self.node_types = list(num_nodes_dict.keys())
        self.metapaths = list(metapaths)
        self.num_nodes_dict = num_nodes_dict
        self.embedding_dim = embedding_dim
        self.use_proximity_loss = use_proximity_loss
        self.neg_sampling_ratio = neg_sampling_ratio

        # Computes beta
        self.conv = torch.nn.ModuleDict(
            {node_type: torch.nn.Conv1d(
                in_channels=node_attr_shape[node_type] if node_type in node_attr_shape else self.embedding_dim,
                out_channels=self.get_relation_size(node_type),
                kernel_size=1) \
                for node_type in self.node_types})

        self.linear = torch.nn.ModuleDict(
            {node_type: torch.nn.Linear(in_channels, embedding_dim, bias=False) \
             for node_type, in_channels in node_attr_shape.items()}
        )
        self.attn_l = torch.nn.ModuleList(
            [torch.nn.Linear(embedding_dim, 1, bias=True) for metapath in self.metapaths])
        self.attn_r = torch.nn.ModuleList(
            [torch.nn.Linear(embedding_dim, 1, bias=True) for metapath in self.metapaths])

        # If some node type are not attributed, assign h_1 embeddings for them
        if node_attr_shape.keys() < num_nodes_dict.keys():
            self.embeddings = torch.nn.ModuleDict(
                {node_type: nn.Embedding(num_embeddings=self.num_nodes_dict[node_type], embedding_dim=embedding_dim) \
                 for node_type in (num_nodes_dict.keys() - node_attr_shape.keys())}
            )
        self.reset_parameters()

    def get_head_relations(self, head_node_type) -> list:
        relations = [metapath for metapath in self.metapaths if metapath[0] == head_node_type]
        return relations

    def get_relation_size(self, node_type) -> int:
        relations = self.get_head_relations(node_type)
        return 1 + len(relations)

    def reset_parameters(self):
        for i, metapath in enumerate(self.metapaths):
            glorot(self.attn_l[i].weight)
            glorot(self.attn_r[i].weight)

        for node_type in self.linear:
            glorot(self.linear[node_type].weight)
        for node_type in self.conv:
            glorot(self.conv[node_type].weight)
        for node_type in self.embeddings:
            self.embeddings[node_type].reset_parameters()

    def __repr__(self):
        return '{}(linear={}, attn={}, embedding={})'.format(self.__class__.__name__,
                                                             {nodetype: linear.weight.shape for
                                                              nodetype, linear in self.linear.items()},
                                                             {metapath: attn.shape for metapath, attn in
                                                              self.attn_l.items()},
                                                             self.embedding_dim)

    def forward(self, x_dict, x_index_dict, edge_index_dict):
        """

        :param x_dict: a dict of node attributes indexed node_type
        :param x_index_dict: a dict of index values indexed by node_type in this mini-batch sampling
        :param edge_index_dict: a dict of edge_index indexed by metapath
        :return: output_emb, loss
        """
        h_dict = {}
        for node_type in self.node_types:
            if node_type in x_dict.keys():
                h_dict[node_type] = (self.linear[node_type](x_dict[node_type])).view(-1, self.embedding_dim)
            else:
                h_dict[node_type] = self.embeddings[node_type].weight[x_index_dict[node_type]]

        beta = {}
        for node_type in self.node_types:
            if node_type in x_dict:
                beta[node_type] = self.conv[node_type].forward(x_dict[node_type].unsqueeze(-1))
            else:
                beta[node_type] = self.conv[node_type].forward(h_dict[node_type].unsqueeze(-1))
            beta[node_type] = torch.softmax(beta[node_type], dim=1)

        if not self.training:
            self._beta = {}
            for node_type in self.node_types:
                _beta = beta[node_type].mean(dim=0).squeeze(-1).cpu().numpy()
                self._beta[node_type] = {metapath: _beta[i] for i, metapath in
                                         enumerate(self.get_head_relations(node_type) + ["self"])}

        score_l, score_r = {}, {}
        for i, metapath in enumerate(self.metapaths):
            head_type, tail_type = metapath[0], metapath[-1]
            score_l[metapath] = self.attn_l[i].forward(h_dict[head_type]).sum(dim=-1)
            score_r[metapath] = self.attn_r[i].forward(h_dict[tail_type]).sum(dim=-1)

        emb_relation_agg = {}
        emb_output = {}
        for node_type in self.node_types:
            emb_relation_agg[node_type] = torch.zeros(
                size=(x_index_dict[node_type].size(0),  # (num_nodes, num_relations, embedding_dim)
                      self.get_relation_size(node_type),
                      self.embedding_dim),
                device=self.conv[node_type].weight.device)

            for i, metapath in enumerate(self.get_head_relations(node_type)):
                if metapath not in edge_index_dict or edge_index_dict[metapath] == None:
                    continue
                head_type, tail_type = metapath[0], metapath[-1]
                head_num_node, tail_num_node = len(x_index_dict[head_type]), len(x_index_dict[tail_type])

                emb_relation_agg[head_type][:, i] = self.propagate(
                    edge_index_dict[metapath],
                    size=(tail_num_node, head_num_node),
                    x=(h_dict[tail_type], h_dict[head_type]),
                    alpha=(score_r[metapath], score_l[metapath]))

            emb_relation_agg[head_type][:, -1] = h_dict[head_type]
            emb_output[node_type] = torch.matmul(emb_relation_agg[head_type].permute(0, 2, 1),
                                                 beta[head_type]).squeeze(-1)

        if self.use_proximity_loss:
            proximity_loss = self.proximity_loss(edge_index_dict, score_l, score_r, x_index_dict)
        else:
            proximity_loss = None

        return emb_output, proximity_loss

    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        # alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index=index, num_nodes=size_i)
        # alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    @staticmethod
    def negative_sample(edge_index, M: int, N: int, num_neg_samples: int):
        num_neg_samples = min(num_neg_samples,
                              M * N - edge_index.size(1))

        rng = range(M * N)
        idx = (edge_index[0] * N + edge_index[1]).to('cpu')  # idx = N * i + j

        perm = torch.tensor(random.sample(rng, num_neg_samples))
        mask = torch.from_numpy(np.isin(perm, idx)).to(torch.bool)
        rest = mask.nonzero().view(-1)
        while rest.numel() > 0:  # pragma: no cover
            tmp = torch.tensor(random.sample(rng, rest.size(0)))
            mask = torch.from_numpy(np.isin(tmp, idx)).to(torch.bool)
            perm[rest] = tmp
            rest = rest[mask.nonzero().view(-1)]

        row = perm / N
        col = perm % N
        neg_edge_index = torch.stack([row, col], dim=0).long()

        return neg_edge_index.to(edge_index.device)

    def proximity_loss(self, edge_index_dict, score_l, score_r, x_index_dict):
        loss = torch.tensor(0, dtype=torch.float, device=self.conv[self.node_types[0]].weight.device)

        # KL Divergence over observed edges, -\sum_(a_ij) a_ij log(e_ij)
        for metapath, edge_index in edge_index_dict.items():
            if edge_index is None: continue
            e_ij = score_l[metapath][edge_index[0]] + score_r[metapath][edge_index[1]]
            loss += -torch.mean(torch.log(torch.sigmoid(e_ij)), dim=-1)

        # KL Divergence over negative sampling edges, -\sum_(a'_uv) a_uv log(-e'_uv)
        for metapath, edge_index in edge_index_dict.items():
            if edge_index is None: continue
            neg_edge_index = self.negative_sample(edge_index,
                                                  M=x_index_dict[metapath[0]].size(0),
                                                  N=x_index_dict[metapath[-1]].size(0),
                                                  num_neg_samples=edge_index.size(1))
            e_ij = score_l[metapath][neg_edge_index[0]] + score_r[metapath][neg_edge_index[1]]
            loss += -torch.mean(torch.log(torch.sigmoid(-e_ij)), dim=-1)

        return loss
