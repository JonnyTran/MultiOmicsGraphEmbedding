import copy
import numpy as np
import pandas as pd
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

from moge.module.sampling import negative_sample, negative_sample_head_tail
from .utils import preprocess_input

class LATTE(nn.Module):
    def __init__(self, in_channels_dict: dict, embedding_dim: int, t_order: int, num_nodes_dict: dict, metapaths: list,
                 activation: str = "relu", attn_heads=1, attn_activation="sharpening", attn_dropout=0.5,
                 use_proximity_loss=True,
                 neg_sampling_ratio=2.0):
        super(LATTE, self).__init__()
        self.metapaths = metapaths
        self.node_types = list(num_nodes_dict.keys())
        self.embedding_dim = embedding_dim * t_order
        self.use_proximity_loss = use_proximity_loss
        self.t_order = t_order
        self.neg_sampling_ratio = neg_sampling_ratio

        layers = []
        t_order_metapaths = copy.deepcopy(metapaths)
        for t in range(t_order):
            if t == 0:
                layers.append(
                    LATTELayer(embedding_dim=embedding_dim, node_attr_shape=in_channels_dict,
                               num_nodes_dict=num_nodes_dict, metapaths=t_order_metapaths, activation=activation,
                               attn_heads=attn_heads, attn_activation=attn_activation, attn_dropout=attn_dropout,
                               use_proximity_loss=use_proximity_loss,
                               neg_sampling_ratio=neg_sampling_ratio \
                                   if isinstance(neg_sampling_ratio, (int, float)) else neg_sampling_ratio[t],
                               first=True))
            else:
                layers.append(
                    LATTELayer(embedding_dim=embedding_dim, node_attr_shape=in_channels_dict,
                               num_nodes_dict=num_nodes_dict, metapaths=t_order_metapaths, activation=activation,
                               attn_heads=attn_heads, attn_activation=attn_activation, attn_dropout=attn_dropout,
                               use_proximity_loss=use_proximity_loss,
                               neg_sampling_ratio=neg_sampling_ratio \
                                   if isinstance(neg_sampling_ratio, (int, float)) else neg_sampling_ratio[t],
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
            edge_values = torch.ones(edge_index_tup.size(1), dtype=torch.float, device=edge_index.device)
        else:
            return None, None  # Should raise an exception

        return edge_index, edge_values

    @staticmethod
    def join_edge_indexes(edge_index_dict_A, edge_index_dict_B, global_node_idx):
        output_dict = {}
        for metapath_a, edge_index_a in edge_index_dict_A.items():
            if is_negative(metapath_a): continue
            edge_index_a, values_a = LATTE.get_edge_index_values(edge_index_a)
            if edge_index_a is None: continue

            for metapath_b, edge_index_b in edge_index_dict_B.items():
                if is_negative(metapath_b): continue
                if metapath_a[-1] != metapath_b[0]: continue

                metapath_join = metapath_a + metapath_b[1:]
                edge_index_b, values_b = LATTE.get_edge_index_values(edge_index_b)
                if edge_index_b is None: continue
                try:
                    new_edge_index = adamic_adar(indexA=edge_index_a, valueA=values_a, indexB=edge_index_b,
                                                 valueB=values_b,
                                                 m=global_node_idx[metapath_a[0]].size(0),
                                                 k=global_node_idx[metapath_a[-1]].size(0),
                                                 n=global_node_idx[metapath_b[-1]].size(0),
                                                 coalesced=True)

                    if new_edge_index[0].size(1) <= 1: continue
                    output_dict[metapath_join] = new_edge_index

                except Exception as e:
                    print(f"{str(e)} \n {metapath_a}: {edge_index_a.size(1)}, {metapath_b}: {edge_index_b.size(1)}")
                    continue

        return output_dict

    def forward(self, x_dict: dict, global_node_idx: dict, edge_index_dict: dict, save_betas=False):
        """
        This
        :param x_dict: Dict of <node_type>:<tensor size (batch_size, in_channels)>
        :param global_node_idx: Dict of <node_type>:<int tensor size (batch_size,)>
        :param edge_index_dict: Dict of <metapath>:<tensor size (2, num_edge_index)>
        :param save_betas: whether to save _beta values for batch
        :return embedding_output, proximity_loss, edge_pred_dict:
        """
        device = global_node_idx[list(global_node_idx.keys())[0]].device
        proximity_loss = torch.tensor(0.0, device=device) if self.use_proximity_loss else None

        h_layers = {node_type: [] for node_type in global_node_idx}
        for t in range(self.t_order):
            if t == 0:
                h_dict, t_proximity_loss, edge_pred_dict = self.layers[t].forward(
                    x_dict=x_dict, global_node_idx=global_node_idx, edge_index_dict=edge_index_dict,
                    save_betas=save_betas)

                if self.t_order >= 2:
                    t_order_edge_index_dict = LATTE.join_edge_indexes(edge_index_dict, edge_index_dict, global_node_idx)
            else:
                h_dict, t_proximity_loss, _ = self.layers[t].forward(
                    x_dict=x_dict, global_node_idx=global_node_idx, edge_index_dict=t_order_edge_index_dict,
                    h1_dict=h_dict, save_betas=save_betas)

                # Only needed if there is a next t-order
                if t < self.t_order - 1:
                    t_order_edge_index_dict = LATTE.join_edge_indexes(t_order_edge_index_dict, edge_index_dict,
                                                                      global_node_idx)

            for node_type in global_node_idx:
                h_layers[node_type].append(h_dict[node_type])

            if self.use_proximity_loss:
                proximity_loss += t_proximity_loss

        embedding_output = {node_type: torch.cat(h_emb_list, dim=1) \
                            for node_type, h_emb_list in h_layers.items() if len(h_emb_list) > 0}

        return embedding_output, proximity_loss, edge_pred_dict

    def get_attn_activation_weights(self, t):
        return dict(zip(self.layers[t].metapaths, self.layers[t].alpha_activation.detach().numpy().tolist()))

    def get_relation_weights(self, t):
        return self.layers[t].get_relation_weights()


class LATTELayer(MessagePassing, pl.LightningModule):
    def __init__(self, embedding_dim: int, node_attr_shape: {str: int}, num_nodes_dict: {str: int}, metapaths: list,
                 activation: str = "relu", attn_heads=4, attn_activation="sharpening", attn_dropout=0.5,
                 use_proximity_loss=True,
                 neg_sampling_ratio=1.0, first=True) -> None:
        super(LATTELayer, self).__init__(aggr="add", flow="target_to_source", node_dim=0)
        self.first = first
        self.node_types = list(num_nodes_dict.keys())
        self.metapaths = list(metapaths)
        self.num_nodes_dict = num_nodes_dict
        self.embedding_dim = embedding_dim
        self.use_proximity_loss = use_proximity_loss
        self.neg_sampling_ratio = neg_sampling_ratio
        self.attn_heads = attn_heads
        self.attn_dropout = attn_dropout

        self.activation = activation.lower()
        if self.activation not in ["sigmoid", "tanh", "relu"]:
            print(f"Embedding activation arg `{self.activation}` did not match, so uses linear activation.")

        self.conv = torch.nn.ModuleDict(
            {node_type: torch.nn.Conv1d(
                in_channels=node_attr_shape[
                    node_type] if self.first and node_type in node_attr_shape else self.embedding_dim,
                out_channels=self.num_head_relations(node_type),
                kernel_size=1) \
                for node_type in self.node_types})  # W_phi.shape (D x F)

        self.linear = torch.nn.ModuleDict(
            {node_type: torch.nn.Linear(node_attr_shape, embedding_dim, bias=True) \
             for node_type, in_channels in node_attr_shape.items()})  # W.shape (F x D_m)

        assert embedding_dim % attn_heads == 0
        self.out_channels = self.embedding_dim // self.attn_heads
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

        # If some node type are not attributed, assign embeddings for them
        non_attr_node_types = (num_nodes_dict.keys() - node_attr_shape.keys())
        if len(non_attr_node_types) > 0:
            if embedding_dim > 256 or sum([v for k, v in self.num_nodes_dict.items()]) > 1000000:
                print("Embedding.device = 'cpu'")
                self.embeddings = {node_type: nn.Embedding(num_embeddings=self.num_nodes_dict[node_type],
                                                           embedding_dim=embedding_dim,
                                                           sparse=False).cpu() for node_type in non_attr_node_types}
            else:
                self.embeddings = torch.nn.ModuleDict(
                    {node_type: nn.Embedding(num_embeddings=self.num_nodes_dict[node_type],
                                             embedding_dim=embedding_dim,
                                             sparse=False) for node_type in non_attr_node_types})
        else:
            self.embeddings = None

        self.reset_parameters()

    def reset_parameters(self):
        for i, metapath in enumerate(self.metapaths):
            glorot(self.attn_l[i])
            glorot(self.attn_r[i])

        # glorot(self.attn_q[-1].weight)

        for node_type in self.linear:
            glorot(self.linear[node_type].weight)
        for node_type in self.conv:
            glorot(self.conv[node_type].weight)

        if self.embeddings is not None and len(self.embeddings.keys()) > 0:
            for node_type in self.embeddings:
                self.embeddings[node_type].reset_parameters()

    def forward(self, x_dict, global_node_idx, edge_index_dict, h1_dict=None, save_betas=False):
        """

        :param x_dict: a dict of node attributes indexed node_type
        :param global_node_idx: A dict of index values indexed by node_type in this mini-batch sampling
        :param edge_index_dict: Sparse adjacency matrices for each metapath relation. A dict of edge_index indexed by metapath
        :param h1_dict: Context embedding of the previous order, required for t >= 2. Default: None (if first order). A dict of (node_type: tensor)
        :return: output_emb, loss
        """
        # H_t = W_t * x
        h_dict = self.get_h_dict(x_dict, global_node_idx)

        # Compute relations attention coefficients
        beta = self.get_beta_weights(x_dict, h_dict, h1_dict, global_node_idx)
        # Save beta weights from testing samples
        if save_betas: self.save_relation_weights(beta, global_node_idx)

        # Compute node-level attention coefficients
        alpha_l, alpha_r = self.get_alphas(edge_index_dict, h_dict, h1_dict)

        # For each metapath in a node_type, use GAT message passing to aggregate h_j neighbors
        out = {}
        for node_type in global_node_idx:
            # Initialize embeddings, size: (num_nodes, num_relations, embedding_dim)
            out[node_type] = self.agg_relation_neighbors(node_type, alpha_l, alpha_r, h_dict, edge_index_dict,
                                                         global_node_idx)
            if self.first:
                out[node_type][:, -1] = h_dict[node_type].view(-1, self.embedding_dim)
            else:
                out[node_type][:, -1] = h1_dict[node_type].view(-1, self.embedding_dim)

            # Soft-select the relation-specific embeddings by a weighted average with beta[node_type]
            out[node_type] = torch.matmul(out[node_type].permute(0, 2, 1), beta[node_type]).squeeze(-1)
            # out[node_type] = out[node_type].mean(dim=1)

            # Apply \sigma activation to all embeddings
            out[node_type] = self.embedding_activation(out[node_type])

        proximity_loss, edge_pred_dict = None, None
        if self.use_proximity_loss:
            proximity_loss, edge_pred_dict = self.proximity_loss(edge_index_dict,
                                                                 alpha_l=alpha_l, alpha_r=alpha_r,
                                                                 global_node_idx=global_node_idx)
        return out, proximity_loss, edge_pred_dict

    def agg_relation_neighbors(self, node_type, alpha_l, alpha_r, h_dict, edge_index_dict, global_node_idx):
        emb_relations = torch.zeros(
            size=(global_node_idx[node_type].size(0),
                  self.num_head_relations(node_type),
                  self.embedding_dim)).type_as(self.conv[node_type].weight)

        for i, metapath in enumerate(self.get_head_relations(node_type)):
            if metapath not in edge_index_dict or edge_index_dict[metapath] == None: continue
            head, tail = metapath[0], metapath[-1]
            num_node_head, num_node_tail = len(global_node_idx[head]), len(global_node_idx[tail])

            edge_index, _ = LATTE.get_edge_index_values(edge_index_dict[metapath])
            if edge_index is None: continue

            # Propapate flows from target nodes to source nodes
            out = self.propagate(
                edge_index=edge_index,
                x=(h_dict[tail], h_dict[head]),
                alpha=(alpha_r[metapath], alpha_l[metapath]),
                size=(num_node_tail, num_node_head),
                metapath_idx=self.metapaths.index(metapath))
            emb_relations[:, i] = out.view(-1, self.embedding_dim)

        return emb_relations

    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i, metapath_idx):
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        # alpha = self.attn_q.forward(torch.cat([alpha_i, alpha_j], dim=1))
        alpha = self.attn_activation(alpha, metapath_idx)
        alpha = softmax(alpha, index=index, ptr=ptr, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.attn_dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def get_h_dict(self, x_dict, global_node_idx):
        h_dict = {}
        for node_type in global_node_idx:
            if node_type in x_dict:
                h_dict[node_type] = self.linear[node_type].forward(x_dict[node_type])
            else:
                # Should only go here in first layer
                h_dict[node_type] = self.embeddings[node_type].weight[global_node_idx[node_type]] \
                    .to(self.conv[node_type].weight.device)

            h_dict[node_type] = h_dict[node_type].view(-1, self.attn_heads, self.out_channels)
        return h_dict

    def get_alphas(self, edge_index_dict, h_dict, h1_dict):
        alpha_l, alpha_r = {}, {}

        for i, metapath in enumerate(self.metapaths):
            if metapath not in edge_index_dict or edge_index_dict[metapath] is None: continue

            head, tail = metapath[0], metapath[-1]
            if self.first:
                alpha_l[metapath] = (h_dict[head] * self.attn_l[i]).sum(dim=-1)
            else:
                alpha_l[metapath] = (h1_dict[head].view(-1, self.attn_heads, self.out_channels) * self.attn_l[i]).sum(
                    dim=-1)

            alpha_r[metapath] = (h_dict[tail] * self.attn_r[i]).sum(dim=-1)

        return alpha_l, alpha_r

    def get_beta_weights(self, x_dict, h_dict, h1_dict, global_node_idx):
        beta = {}
        for node_type in global_node_idx:
            if self.first:
                if node_type in x_dict:
                    beta[node_type] = self.conv[node_type].forward(x_dict[node_type].unsqueeze(-1))
                else:
                    # node_type is not attributed, use h_dict contains self.embeddings
                    beta[node_type] = self.conv[node_type].forward(
                        h_dict[node_type].view(-1, self.embedding_dim).unsqueeze(-1))
            else:
                beta[node_type] = self.conv[node_type].forward(h1_dict[node_type].unsqueeze(-1))

            beta[node_type] = torch.softmax(beta[node_type], dim=1)
        return beta

    def predict_scores(self, edge_index, alpha_l, alpha_r, metapath, logits=False):
        assert metapath in self.metapaths, f"If metapath `{metapath}` is tag_negative()'ed, then pass it with untag_negative()"

        e_ij = self.attn_q.forward(
            torch.cat([alpha_l[metapath][edge_index[0]], alpha_r[metapath][edge_index[1]]], dim=1)).squeeze(-1)

        if logits:
            return e_ij
        else:
            return F.sigmoid(e_ij)

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
            else:
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

        loss = torch.true_divide(loss, len(edge_index_dict) * 2)
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

    def save_relation_weights(self, beta, global_node_idx):
        # Only save relation weights if beta has weights for all node_types in the global_node_idx batch
        if len(beta) < len(self.node_types): return

        self._betas = {}
        self._beta_avg = {}
        self._beta_std = {}
        for node_type in beta:
            with torch.no_grad():
                self._betas[node_type] = pd.DataFrame(beta[node_type].squeeze(-1).cpu().numpy(),
                                                      columns=self.get_head_relations(node_type) + [node_type, ],
                                                      index=global_node_idx[node_type].cpu().numpy())

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
