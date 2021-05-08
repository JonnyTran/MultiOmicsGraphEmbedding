import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from .latte import LATTE, is_negative, untag_negative, tag_negative
from moge.module.sampling import negative_sample


class LATTEConv(MessagePassing, pl.LightningModule):
    def __init__(self, input_dim: {str: int}, output_dim: int, num_nodes_dict: {str: int}, metapaths: list,
                 activation: str = "relu", layernorm=False, attn_heads=4, attn_activation="sharpening",
                 attn_dropout=0.2, use_proximity=False, neg_sampling_ratio=1.0) -> None:
        super(LATTEConv, self).__init__(aggr="add", flow="target_to_source", node_dim=0)
        self.node_types = list(num_nodes_dict.keys())
        self.metapaths = list(metapaths)
        self.num_nodes_dict = num_nodes_dict
        self.embedding_dim = output_dim
        self.use_proximity = use_proximity
        self.neg_sampling_ratio = neg_sampling_ratio
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

        if layernorm:
            self.layernorm = torch.nn.ModuleDict({
                node_type: nn.LayerNorm(output_dim) \
                for node_type in self.node_types})

        self.conv = torch.nn.ModuleDict(
            {node_type: torch.nn.Conv1d(
                in_channels=input_dim,
                out_channels=self.num_head_relations(node_type),
                kernel_size=1) \
                for node_type in self.node_types})  # W_phi.shape (D x F)

        self.linear_l = nn.ModuleDict(
            {node_type: nn.Linear(input_dim, output_dim, bias=True) \
             for node_type in self.node_types})  # W.shape (F x F)
        self.linear_r = nn.ModuleDict(
            {node_type: nn.Linear(input_dim, output_dim, bias=True) \
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

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.2)
        for i, metapath in enumerate(self.metapaths):
            nn.init.xavier_normal_(self.attn_l[i], gain=gain)
            nn.init.xavier_normal_(self.attn_r[i], gain=gain)

        gain = nn.init.calculate_gain('relu')
        for node_type in self.linear_l:
            nn.init.xavier_normal_(self.linear_l[node_type].weight, gain=gain)
        for node_type in self.linear_r:
            nn.init.xavier_normal_(self.linear_r[node_type].weight, gain=gain)

        for node_type in self.conv:
            nn.init.xavier_normal_(self.conv[node_type].weight, gain=1)

        if self.embeddings is not None and len(self.embeddings.keys()) > 0:
            for node_type in self.embeddings:
                self.embeddings[node_type].reset_parameters()

    def forward(self, x_l, edge_index_dict, global_node_idx, x_r=None, save_betas=False):
        """

        :param x_l: a dict of node attributes indexed node_type
        :param global_node_idx: A dict of index values indexed by node_type in this mini-batch sampling
        :param edge_index_dict: Sparse adjacency matrices for each metapath relation. A dict of edge_index indexed by metapath
        :param x_r: Context embedding of the previous order, required for t >= 2. Default: None (if first order). A dict of (node_type: tensor)
        :return: output_emb, loss
        """
        l_dict = self.get_h_dict(x_l, global_node_idx, left_right="left")
        r_dict = self.get_h_dict(x_r, global_node_idx, left_right="right")

        # Predict relations attention coefficients
        beta = self.get_beta_weights(x_l, global_node_idx=global_node_idx)
        # Save beta weights from testing samples
        if not self.training: self.save_relation_weights(beta, global_node_idx)

        # Compute node-level attention coefficients
        alpha_l, alpha_r = self.get_alphas(edge_index_dict, l_dict, r_dict)

        # For each metapath in a node_type, use GAT message passing to aggregate h_j neighbors
        out = {}
        for ntype in global_node_idx:
            out[ntype] = self.agg_relation_neighbors(node_type=ntype, alpha_l=alpha_l, alpha_r=alpha_r,
                                                     l_dict=l_dict, r_dict=r_dict, edge_index_dict=edge_index_dict,
                                                     global_node_idx=global_node_idx)
            out[ntype][:, -1] = l_dict[ntype].view(-1, self.embedding_dim)

            # Soft-select the relation-specific embeddings by a weighted average with beta[node_type]
            out[ntype] = torch.bmm(out[ntype].permute(0, 2, 1), beta[ntype]).squeeze(-1)

            if hasattr(self, "layernorm"):
                out[ntype] = self.layernorm[ntype](out[ntype])

            if hasattr(self, "activation"):
                out[ntype] = self.activation(out[ntype])

        proximity_loss, edge_pred_dict = None, None
        if self.use_proximity:
            proximity_loss, edge_pred_dict = self.proximity_loss(edge_index_dict,
                                                                 alpha_l=alpha_l, alpha_r=alpha_r,
                                                                 global_node_idx=global_node_idx)
        return out, proximity_loss, edge_pred_dict

    def agg_relation_neighbors(self, node_type, alpha_l, alpha_r, l_dict, r_dict, edge_index_dict, global_node_idx):
        # Initialize embeddings, size: (num_nodes, num_relations, embedding_dim)
        emb_relations = torch.zeros(
            size=(global_node_idx[node_type].size(0),
                  self.num_head_relations(node_type),
                  self.embedding_dim)).type_as(self.conv[node_type].weight)

        for i, metapath in enumerate(self.get_head_relations(node_type)):
            if metapath not in edge_index_dict or edge_index_dict[metapath] == None: continue
            head, tail = metapath[0], metapath[-1]
            num_node_head, num_node_tail = global_node_idx[head].size(0), global_node_idx[tail].size(0)

            edge_index, values = LATTE.get_edge_index_values(edge_index_dict[metapath])
            if edge_index is None: continue

            # Propapate flows from target nodes to source nodes
            out = self.propagate(
                edge_index=edge_index,
                x=(r_dict[tail], l_dict[head]),
                alpha=(alpha_r[metapath], alpha_l[metapath]),
                size=(num_node_tail, num_node_head),
                metapath_idx=self.metapaths.index(metapath))
            emb_relations[:, i] = out.view(-1, self.embedding_dim)

        return emb_relations

    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i, metapath_idx):
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = self.attn_activation(alpha, metapath_idx)
        alpha = softmax(alpha, index=index, ptr=ptr, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.attn_dropout, training=self.training)

        return x_j * alpha

    def get_h_dict(self, input, global_node_idx, left_right="left"):
        h_dict = {}
        for node_type in global_node_idx:
            if left_right == "left":
                h_dict[node_type] = self.linear_l[node_type].forward(input[node_type])
            elif left_right == "right":
                h_dict[node_type] = self.linear_r[node_type].forward(input[node_type])

            h_dict[node_type] = h_dict[node_type].view(-1, self.attn_heads, self.out_channels)

        return h_dict

    def get_alphas(self, edge_index_dict, l_dict, r_dict):
        alpha_l, alpha_r = {}, {}

        for i, metapath in enumerate(self.metapaths):
            if metapath not in edge_index_dict or edge_index_dict[metapath] is None:
                continue
            head, tail = metapath[0], metapath[-1]

            alpha_l[metapath] = l_dict[head] * self.attn_l[i]
            alpha_r[metapath] = r_dict[tail] * self.attn_r[i]
        return alpha_l, alpha_r

    def get_beta_weights(self, h_dict, global_node_idx):
        beta = {}
        for node_type in global_node_idx:
            beta[node_type] = self.conv[node_type].forward(h_dict[node_type].unsqueeze(-1))

            beta[node_type] = torch.softmax(beta[node_type], dim=1)
        return beta

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

    def attn_activation(self, alpha, metapath_id):
        if isinstance(self.alpha_activation, torch.Tensor):
            return self.alpha_activation[metapath_id] * alpha
        elif isinstance(self.alpha_activation, nn.Module):
            return self.alpha_activation(alpha)
        else:
            return alpha

    def get_head_relations(self, head_node_type, to_str=False) -> list:
        relations = [".".join(metapath) if to_str and isinstance(metapath, tuple) else metapath for metapath in
                     self.metapaths if
                     metapath[0] == head_node_type]
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
            relations = self.get_head_relations(node_type, True) + [node_type, ]

            with torch.no_grad():
                self._betas[node_type] = pd.DataFrame(beta[node_type].squeeze(-1).cpu().numpy(),
                                                      columns=relations,
                                                      index=global_node_idx[node_type].cpu().numpy())

                _beta_avg = np.around(beta[node_type].mean(dim=0).squeeze(-1).cpu().numpy(), decimals=3)
                _beta_std = np.around(beta[node_type].std(dim=0).squeeze(-1).cpu().numpy(), decimals=2)
                self._beta_avg[node_type] = {metapath: _beta_avg[i] for i, metapath in
                                             enumerate(relations)}
                self._beta_std[node_type] = {metapath: _beta_std[i] for i, metapath in
                                             enumerate(relations)}

    def save_attn_weights(self, node_type, attn_weights, node_idx):
        if not hasattr(self, "_betas"):
            self._betas = {}
        if not hasattr(self, "_beta_avg"):
            self._beta_avg = {}
        if not hasattr(self, "_beta_std"):
            self._beta_std = {}

        betas = attn_weights.sum(1)

        relations = self.get_head_relations(node_type, True) + [node_type, ]

        with torch.no_grad():
            self._betas[node_type] = pd.DataFrame(betas.cpu().numpy(),
                                                  columns=relations,
                                                  index=node_idx.cpu().numpy())

            _beta_avg = np.around(betas.mean(dim=0).cpu().numpy(), decimals=3)
            _beta_std = np.around(betas.std(dim=0).cpu().numpy(), decimals=2)
            self._beta_avg[node_type] = {metapath: _beta_avg[i] for i, metapath in
                                         enumerate(relations)}
            self._beta_std[node_type] = {metapath: _beta_std[i] for i, metapath in
                                         enumerate(relations)}

    def get_relation_weights(self):
        """
        Get the mean and std of relation attention weights for all nodes
        :return:
        """
        return {(metapath if "." in metapath or len(metapath) > 1 else node_type): (avg, std) \
                for node_type in self._beta_avg for (metapath, avg), (relation_b, std) in
                zip(self._beta_avg[node_type].items(), self._beta_std[node_type].items())}
