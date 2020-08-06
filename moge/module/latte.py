import copy
import numpy as np
import torch
from torch import nn as nn

import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax
import torch_sparse
from torch_sparse.tensor import SparseTensor
from torch_sparse.matmul import matmul
import pytorch_lightning as pl

from moge.module.sampling import negative_sample
from .utils import preprocess_input

class LATTE(nn.Module):
    def __init__(self, in_channels_dict: dict, embedding_dim: int, t_order: int, num_nodes_dict: dict, metapaths: list,
                 activation: str = "relu", use_proximity_loss=True, neg_sampling_ratio=2.0):
        super(LATTE, self).__init__()
        self.metapaths = metapaths
        self.node_types = list(num_nodes_dict.keys())
        self.embedding_dim = embedding_dim * t_order
        self.use_proximity_loss = use_proximity_loss
        self.t_order = t_order
        self.neg_sampling_ratio = neg_sampling_ratio
        self.compute_device = "cuda:3"

        layers = []
        t_order_metapaths = copy.deepcopy(metapaths)
        for t in range(t_order):
            if t == 0:
                layers.append(
                    LATTELayer(embedding_dim=embedding_dim, node_attr_shape=in_channels_dict,
                               num_nodes_dict=num_nodes_dict, metapaths=t_order_metapaths,
                               activation=activation,
                               use_proximity_loss=use_proximity_loss, neg_sampling_ratio=neg_sampling_ratio,
                               first=True))
            else:
                layers.append(
                    LATTELayer(embedding_dim=embedding_dim, node_attr_shape=in_channels_dict,
                               num_nodes_dict=num_nodes_dict, metapaths=t_order_metapaths,
                               activation=activation,
                               use_proximity_loss=use_proximity_loss, neg_sampling_ratio=neg_sampling_ratio,
                               first=False))
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
        elif isinstance(edge_index_tup, torch.Tensor) and edge_index_tup.size(1) > 1:
            edge_index = edge_index_tup
            edge_values = torch.ones(edge_index_tup.size(1), dtype=torch.float, device=edge_index_tup.device)
        else:
            return None, None  # Should raise an exception

        return edge_index, edge_values

    @staticmethod
    def join_edge_indexes(edge_index_dict_A, edge_index_dict_B, global_node_idx):
        output_dict = {}
        for metapath_a, edge_index_a in edge_index_dict_A.items():
            if edge_index_a is None or (isinstance(edge_index_a, tuple) and edge_index_a[0] is None): continue
            edge_index_a, values_a = LATTE.get_edge_index_values(edge_index_a)
            if edge_index_a is None: continue

            for metapath_b, edge_index_b in edge_index_dict_B.items():
                if edge_index_b is None or (isinstance(edge_index_b, tuple) and edge_index_b[0] is None): continue

                if metapath_a[-1] == metapath_b[0]:
                    metapath_join = metapath_a + metapath_b[1:]
                    edge_index_b, values_b = LATTE.get_edge_index_values(edge_index_b)
                    if edge_index_b is None: continue
                    try:
                        new_edge_index = torch_sparse.spspmm(indexA=edge_index_a, valueA=values_a,
                                                             indexB=edge_index_b, valueB=values_b,
                                                             m=global_node_idx[metapath_a[0]].size(0),
                                                             k=global_node_idx[metapath_a[-1]].size(0),
                                                             n=global_node_idx[metapath_b[-1]].size(0),
                                                             coalesced=True)

                        if new_edge_index[0].size(1) <= 5: continue
                        output_dict[metapath_join] = new_edge_index

                    except Exception as e:
                        print(f"{metapath_a}, {edge_index_a.size(1)}", f"{metapath_b}, {edge_index_b.size(1)}")
                        print("edge_index_a", edge_index_a.size(1), edge_index_a.dtype)
                        print("values_a", values_a.size(), values_a.dtype)
                        print("edge_index_b", edge_index_b.size(1), edge_index_b.dtype)
                        print("values_b", values_b.size(), values_b.dtype)
                        # raise e
                        continue

        return output_dict

    def forward(self, x_dict, global_node_idx, edge_index_dict):
        proximity_loss = torch.tensor(0.0, device=self.layers[0].device) if self.use_proximity_loss else None

        h_all_dict = {node_type: [] for node_type in global_node_idx}
        for t in range(self.t_order):
            if t == 0:
                h_dict, t_proximity_loss, edge_pred_dict = self.layers[t].forward(x_dict=x_dict,
                                                                                  global_node_idx=global_node_idx,
                                                                                  edge_index_dict=edge_index_dict)
                if self.t_order >= 2:
                    with torch.no_grad():
                        t_order_edge_index_dict = LATTE.join_edge_indexes(
                            preprocess_input(edge_index_dict, device=self.compute_device),
                            preprocess_input(edge_index_dict, device=self.compute_device),
                            preprocess_input(global_node_idx, device=self.compute_device))
            else:
                h_dict, t_proximity_loss, edge_pred_dict_t = self.layers[t].forward(
                    x_dict=x_dict, global_node_idx=global_node_idx,
                    edge_index_dict=t_order_edge_index_dict,
                    h1_dict=h_dict)
                # h1_dict={node_type: h_emb.detach() for node_type, h_emb in
                #          h_dict.items()})  # Detach the prior-order embeddings from backprop gradients

                if edge_pred_dict is not None and edge_pred_dict_t:
                    edge_pred_dict.update(edge_pred_dict_t)

                with torch.no_grad():
                    t_order_edge_index_dict = LATTE.join_edge_indexes(
                        preprocess_input(t_order_edge_index_dict, device=self.compute_device),
                        preprocess_input(edge_index_dict, device=self.compute_device),
                        preprocess_input(global_node_idx, device=self.compute_device))

            for node_type in global_node_idx:
                h_all_dict[node_type].append(h_dict[node_type])
            if self.use_proximity_loss:
                proximity_loss += t_proximity_loss

        embedding_output = {node_type: torch.cat(h_emb_list, dim=1) \
                            for node_type, h_emb_list in h_all_dict.items() if len(h_emb_list) > 0}

        return embedding_output, proximity_loss, edge_pred_dict


class LATTELayer(MessagePassing, pl.LightningModule):
    RELATIONS_DIM = 1

    def __init__(self, embedding_dim: int, node_attr_shape: {str: int}, num_nodes_dict: {str: int}, metapaths: list,
                 activation: str = "relu", use_proximity_loss=True, neg_sampling_ratio=1.0, first=True) -> None:
        super(LATTELayer, self).__init__(aggr="add", flow="target_to_source", node_dim=0)
        self.first = first
        self.node_types = list(num_nodes_dict.keys())
        self.metapaths = list(metapaths)
        self.num_nodes_dict = num_nodes_dict
        self.embedding_dim = embedding_dim
        self.use_proximity_loss = use_proximity_loss
        self.neg_sampling_ratio = neg_sampling_ratio

        self.activation = activation
        assert self.activation in ["sigmoid", "tanh", "relu", None]

        self.conv = torch.nn.ModuleDict(
            {node_type: torch.nn.Conv1d(
                in_channels=node_attr_shape[
                    node_type] if self.first and node_type in node_attr_shape else self.embedding_dim,
                out_channels=self.num_head_relations(node_type),
                kernel_size=1) \
                for node_type in self.node_types})  # W_phi.shape (H_-1 x F)

        self.linear = torch.nn.ModuleDict(
            {node_type: torch.nn.Linear(in_channels, embedding_dim, bias=True) \
             for node_type, in_channels in node_attr_shape.items()})  # W.shape (F x D_m)
        self.attn_l = torch.nn.ModuleList(
            [torch.nn.Linear(embedding_dim, 1, bias=True) for metapath in self.metapaths])
        self.attn_r = torch.nn.ModuleList(
            [torch.nn.Linear(embedding_dim, 1, bias=True) for metapath in self.metapaths])
        self.alpha_weights = nn.Parameter(torch.Tensor(len(self.metapaths)).fill_(1.0))

        # If some node type are not attributed, assign embeddings for them
        non_attr_node_types = (num_nodes_dict.keys() - node_attr_shape.keys())
        if len(non_attr_node_types) > 0:
            self.embeddings = torch.nn.ModuleDict(
                {node_type: nn.Embedding(num_embeddings=self.num_nodes_dict[node_type],
                                         embedding_dim=embedding_dim,
                                         sparse=False).cuda(2) for node_type in non_attr_node_types})
        else:
            self.embeddings = None

        self.reset_parameters()

    def reset_parameters(self):
        for i, metapath in enumerate(self.metapaths):
            glorot(self.attn_l[i].weight)
            glorot(self.attn_r[i].weight)

        for node_type in self.linear:
            glorot(self.linear[node_type].weight)
        for node_type in self.conv:
            glorot(self.conv[node_type].weight)

        if self.embeddings is not None and len(self.embeddings.keys()) > 0:
            for node_type in self.embeddings:
                self.embeddings[node_type].reset_parameters()

    def get_head_relations(self, head_node_type) -> list:
        relations = [metapath for metapath in self.metapaths if metapath[0] == head_node_type]
        return relations

    def num_head_relations(self, node_type) -> int:
        """
        Return the number of metapaths with head node type equals to :param node_type: and plus one for none-selection.
        :param node_type (str):
        :return:
        """
        relations = self.get_head_relations(node_type)
        return len(relations) + 1

    def save_relation_weights(self, beta):
        # Only save relation weights if beta has weights for all node_types in the global_node_idx batch
        if len(beta) < len(self.node_types): return
        self._beta_avg = {}
        self._beta_std = {}
        for node_type in beta:
            _beta_avg = np.around(beta[node_type].mean(dim=0).squeeze(-1).cpu().numpy(), decimals=3)
            _beta_std = np.around(beta[node_type].std(dim=0).squeeze(-1).cpu().numpy(), decimals=2)
            self._beta_avg[node_type] = {metapath: _beta_avg[i] for i, metapath in
                                         enumerate(self.get_head_relations(node_type) + ["self"])}
            self._beta_std[node_type] = {metapath: _beta_std[i] for i, metapath in
                                         enumerate(self.get_head_relations(node_type) + ["self"])}

    def get_relation_weights(self):
        """
        Get the mean and std of relation attention weights for all nodes in testing/validation steps
        :return:
        """
        return {".".join(relation) if isinstance(relation, tuple) else node_type: (avg, std) \
                for node_type in self._beta_avg for (relation, avg), (relation_b, std) in
                zip(self._beta_avg[node_type].items(), self._beta_std[node_type].items())}

    def embedding_activation(self, embeddings):
        if self.activation == "sigmoid":
            return F.sigmoid(embeddings)
        elif self.activation == "tanh":
            return F.tanh(embeddings)
        elif self.activation == "relu":
            return F.relu(embeddings)
        else:
            return embeddings

    def forward(self, x_dict, global_node_idx, edge_index_dict, h1_dict=None):
        """

        :param x_dict: a dict of node attributes indexed node_type
        :param global_node_idx: A dict of index values indexed by node_type in this mini-batch sampling
        :param edge_index_dict: Sparse adjacency matrices for each metapath relation. A dict of edge_index indexed by metapath
        :param h1_dict: Embeddings of the previous order. Default: None (if first order). A dict of edge_index indexed by metapath
        :return: output_emb, loss
        """
        # H_t = W_t * x
        h_dict = self.get_h_dict(x_dict, global_node_idx)

        # Compute relations attention coefficients
        beta = self.get_beta_weights(x_dict, h_dict, h1_dict, global_node_idx)
        # Save beta weights from testing samples
        if not self.training: self.save_relation_weights(beta)

        # Compute node-level attention coefficients
        alpha_l, alpha_r = self.get_alphas(edge_index_dict, h_dict, h1_dict)

        # For each metapath in a node_type, use GAT message passing to aggregate h_j neighbors
        emb_relation_agg = {}
        emb_output = {}
        for node_type in global_node_idx:
            # Initialize embeddings, size: (num_nodes, num_relations, embedding_dim)
            emb_relation_agg[node_type] = torch.zeros(
                size=(global_node_idx[node_type].size(0),
                      self.num_head_relations(node_type),
                      self.embedding_dim)).type_as(self.conv[node_type].weight)

            for i, metapath in enumerate(self.get_head_relations(node_type)):
                if metapath not in edge_index_dict or edge_index_dict[metapath] == None:
                    continue
                head_type, tail_type = metapath[0], metapath[-1]
                num_node_head, num_node_tail = len(global_node_idx[head_type]), len(global_node_idx[tail_type])

                if isinstance(edge_index_dict[metapath], tuple):
                    edge_index, _ = edge_index_dict[metapath]
                else:
                    edge_index = edge_index_dict[metapath]
                if edge_index.size(1) <= 5: continue

                # Propapate flows from target nodes to source nodes
                emb_relation_agg[head_type][:, i] = self.propagate(
                    edge_index=edge_index.to(h_dict[head_type].device),
                    size=(num_node_tail, num_node_head),
                    x=(h_dict[tail_type], h_dict[head_type]),
                    alpha=(alpha_r[metapath], alpha_l[metapath]),
                    metapath_idx=self.metapaths.index(metapath),
                )

            # Assign the "self" embedding representation
            emb_relation_agg[node_type][:, -1] = h_dict[node_type]

            # Soft-select the relation-specific embeddings by a weighted average with beta[node_type]
            emb_output[node_type] = torch.matmul(emb_relation_agg[node_type].permute(0, 2, 1),
                                                 beta[node_type]).squeeze(-1)
            # emb_output[node_type] = emb_relation_agg[node_type].mean(dim=1) # average over all relations
            # Apply \sigma activation to all embeddings
            emb_output[node_type] = self.embedding_activation(emb_output[node_type])

        if self.use_proximity_loss:
            proximity_loss, edge_pred_dict = self.proximity_loss(
                preprocess_input(edge_index_dict, device=h_dict[head_type].device),
                alpha_l=alpha_l,
                alpha_r=alpha_r,
                global_node_idx=global_node_idx)

        else:
            proximity_loss = None
            edge_pred_dict = None

        return emb_output, proximity_loss, edge_pred_dict

    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i, metapath_idx):
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = self.alpha_weights[metapath_idx] * alpha
        alpha = softmax(alpha, index=index, ptr=ptr, num_nodes=size_i)
        alpha = F.dropout(alpha, p=0.25, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def get_h_dict(self, x_dict, global_node_idx):
        h_dict = {}
        for node_type in global_node_idx:
            if node_type in x_dict:
                h_dict[node_type] = (self.linear[node_type](x_dict[node_type])).view(-1, self.embedding_dim)
            else:
                h_dict[node_type] = self.embeddings[node_type].weight[global_node_idx[node_type]].to(
                    self.conv[node_type].weight.device)
        return h_dict

    def get_beta_weights(self, x_dict, h_dict, h1_dict, global_node_idx):
        beta = {}
        for node_type in global_node_idx:
            if self.first:
                if node_type in x_dict:
                    beta[node_type] = self.conv[node_type].forward(x_dict[node_type].unsqueeze(-1))
                else:
                    # node_type is not attributed, use self.embeddings in first layer
                    beta[node_type] = self.conv[node_type].forward(h_dict[node_type].unsqueeze(-1))
            elif not self.first:
                beta[node_type] = self.conv[node_type].forward(h1_dict[node_type].unsqueeze(-1))
            else:
                raise Exception()
            beta[node_type] = torch.softmax(beta[node_type], dim=1)
        return beta

    def get_alphas(self, edge_index_dict, h_dict, h1_dict):
        alpha_l, alpha_r = {}, {}
        for i, metapath in enumerate(self.metapaths):
            if metapath not in edge_index_dict or edge_index_dict[metapath] is None:
                continue
            head_type, tail_type = metapath[0], metapath[-1]
            if self.first:
                alpha_l[metapath] = self.attn_l[i].forward(h_dict[head_type]).sum(dim=-1)  # alpha_l = attn_l * W * x_1
            else:
                alpha_l[metapath] = self.attn_l[i].forward(h1_dict[head_type]).sum(dim=-1)  # alpha_l = attn_l * h_1

            alpha_r[metapath] = self.attn_r[i].forward(h_dict[tail_type]).sum(dim=-1)  # alpha_r = attn_r * W * x_1
        return alpha_l, alpha_r

    def predict_scores(self, edge_index, alpha_l, alpha_r, metapath, logits=False):
        e_ij = alpha_l[metapath][edge_index[0]] + alpha_r[metapath][edge_index[1]]
        if logits:
            return e_ij
        else:
            return F.sigmoid(e_ij)

    def proximity_loss(self, edge_index_dict, alpha_l, alpha_r, global_node_idx):
        loss = torch.tensor(0.0, dtype=torch.float, device=self.attn_l[0].weight.device)
        edge_pred_dict = {}
        # KL Divergence over observed edges, -\sum_(a_ij) a_ij log(e_ij)
        for metapath, edge_index in edge_index_dict.items():
            if isinstance(edge_index, tuple):  # Weighted edges
                edge_index, values = edge_index
            else:
                values = 1.0
            if edge_index is None: continue

            e_pred = self.predict_scores(edge_index, alpha_l, alpha_r, metapath, logits=False)
            edge_pred_dict[metapath] = e_pred
        loss += -torch.mean(values * torch.log(e_pred), dim=-1)

        # KL Divergence over negative sampling edges, -\sum_(a'_uv) a_uv log(-e'_uv)
        for metapath, edge_index in edge_index_dict.items():
            if isinstance(edge_index, tuple):  # Weighted edges
                edge_index, _ = edge_index
            if edge_index is None or edge_index.size(1) <= 5: continue

            neg_edge_index = negative_sample(edge_index,
                                             M=global_node_idx[metapath[0]].size(0),
                                             N=global_node_idx[metapath[-1]].size(0),
                                             num_neg_samples=edge_index.size(1) * self.neg_sampling_ratio)
            if neg_edge_index.size(1) <= 1: continue

            e_pred = self.predict_scores(neg_edge_index, alpha_l, alpha_r, metapath, logits=True)
            edge_pred_dict[tag_negative(metapath)] = F.sigmoid(e_pred)

            loss += -torch.mean(torch.log(torch.sigmoid(-e_pred)), dim=-1)

        return loss, edge_pred_dict


def tag_negative(metapath):
    if isinstance(metapath, tuple):
        return metapath + ("neg",)
    elif isinstance(metapath, str):
        return metapath + "neg"
    else:
        return "neg"

def adamic_adar(indexA, valueA, indexB, valueB, m, k, n, coalesced=False):
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

    C = matmul(matmul(A, D), B)
    row, col, value = C.coo()

    return torch.stack([row, col], dim=0), value


