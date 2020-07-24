import torch
from torch import nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax


class LATTE(nn.Module):
    def __init__(self, layers: int, embedding_dim: int, num_nodes_dict: dict, node_attr_shape: dict):
        super(LATTE, self).__init__()

        self.layers = nn.ModuleList(layers)

    def forward(self, x_dict, x_index_dict, edge_index_dict):
        pass

    def loss(self):
        pass


class LATTELayer(MessagePassing):
    def __init__(self, t_order: int, embedding_dim: int, num_nodes_dict: dict, node_attr_shape: dict,
                 metapaths: list) -> None:
        super(LATTELayer, self).__init__(aggr="add", flow="source_to_target")
        self.t_order = t_order
        self.node_types = list(num_nodes_dict.keys())
        self.metapaths = list(metapaths)
        self.num_nodes_dict = num_nodes_dict
        self.embedding_dim = embedding_dim

        if self.t_order > 1:
            self.conv = torch.nn.ModuleDict(
                {node_type: torch.nn.Conv1d(in_channels=self.embedding_dim,
                                            out_channels=len(self.get_relations(node_type)),
                                            kernel_size=1) \
                 for node_type in self.node_types})
        else:
            self.conv = torch.nn.ModuleDict(
                {node_type: torch.nn.Conv1d(
                    in_channels=node_attr_shape[node_type] if node_type in node_attr_shape else self.embedding_dim,
                    out_channels=len(self.get_relations(node_type)),
                    kernel_size=1) \
                    for node_type in self.node_types})

        self.linear = torch.nn.ModuleDict(
            {node_type: torch.nn.Linear(in_channels, embedding_dim, bias=False) \
             for node_type, in_channels in node_attr_shape.items()}
        )
        self.attn_l = {edge_type: Parameter(torch.Tensor(1, 2 * embedding_dim)) for edge_type in self.metapaths}
        self.attn_r = {edge_type: Parameter(torch.Tensor(1, 2 * embedding_dim)) for edge_type in self.metapaths}

        if node_attr_shape.keys() < num_nodes_dict.keys():  # If some node type are not attributed
            self.embeddings = torch.nn.ModuleDict(
                {node_type: nn.Embedding(num_embeddings=self.num_nodes_dict[node_type], embedding_dim=embedding_dim) \
                 for node_type in (num_nodes_dict.keys() - node_attr_shape.keys())}
            )
        self.reset_parameters()

    def get_relations(self, head_node_type):
        relations = {metapath for metapath in self.metapaths if metapath[0] == head_node_type}
        return relations

    def reset_parameters(self):
        for metapath in self.attn_l:
            glorot(self.attn_l[metapath])
        for metapath in self.attn_r:
            glorot(self.attn_r[metapath])
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
        for node_type in self.node_types:
            if node_type in x_dict:
                x_dict[node_type] = torch.tanh(self.linear[node_type](x_dict[node_type])).view(-1, self.embedding_dim)
            else:
                x_dict[node_type] = self.embeddings[node_type].weight[x_index_dict[node_type]]

        alpha_l = {node_type: (x_dict[node_type] * self.att_l[node_type]).sum(dim=-1) for node_type in self.node_types}
        alpha_r = {node_type: (x_dict[node_type] * self.att_r[node_type]).sum(dim=-1) for node_type in self.node_types}

        emb = {}
        for node_type in self.node_types:
            emb[node_type] = {}
        for metapath, edge_index in edge_index_dict.items():
            if node_type not in metapath:
                continue
            head_type = metapath[0]
            tail_type = metapath[-1]
            emb[node_type][metapath] = self.propagate(edge_index, h=(), alpha=(), size=None)

        loss_1 = self.loss(edge_index_dict, alpha_l, alpha_r)

        return emb, loss_1

    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    @staticmethod
    def negative_sample(edge_index, m_nodes, n_nodes, num_neg_samples):
        return edge_index

    def loss(self, edge_index_dict, alpha_l, alpha_r):
        for (head, tail), edge_index in edge_index_dict.items():
            pass
