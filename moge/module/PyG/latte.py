import copy

import torch
from torch import nn as nn
from torch_sparse.tensor import SparseTensor

from moge.module.PyG.conv import LATTEConv


class LATTE(nn.Module):
    def __init__(self, n_layers: int, embedding_dim: int, in_channels_dict: dict, num_nodes_dict: dict, metapaths: list,
                 activation: str = "relu", attn_heads=1, attn_activation="sharpening", attn_dropout=0.5,
                 use_proximity=True, neg_sampling_ratio=2.0, edge_sampling=True, cpu_embeddings=False,
                 layer_pooling=False, hparams=None):
        super(LATTE, self).__init__()
        self.metapaths = metapaths
        self.node_types = list(num_nodes_dict.keys())
        self.embedding_dim = embedding_dim * n_layers
        self.use_proximity = use_proximity
        self.t_order = n_layers
        self.neg_sampling_ratio = neg_sampling_ratio
        self.edge_sampling = edge_sampling

        self.layer_pooling = layer_pooling

        # align the dimension of different types of nodes
        self.feature_projection = nn.ModuleDict({
            ntype: nn.Linear(in_channels_dict[ntype], embedding_dim) for ntype in in_channels_dict
        })

        layers = []
        t_order_metapaths = copy.deepcopy(metapaths)
        for t in range(n_layers):
            is_output_layer = (t + 1 == n_layers) and (hparams.nb_cls_dense_size < 0)

            layers.append(
                LATTEConv(input_dim=embedding_dim,
                          output_dim=hparams.n_classes if is_output_layer else embedding_dim,
                          num_nodes_dict=num_nodes_dict,
                          metapaths=t_order_metapaths,
                          activation=None if is_output_layer else activation,
                          layernorm=False if not hasattr(hparams,
                                                         "layernorm") or is_output_layer else hparams.layernorm,
                          attn_heads=attn_heads,
                          attn_activation=attn_activation,
                          attn_dropout=attn_dropout, use_proximity=use_proximity,
                          neg_sampling_ratio=neg_sampling_ratio))
            t_order_metapaths = LATTE.join_metapaths(t_order_metapaths, metapaths)
        self.layers = nn.ModuleList(layers)

        # If some node type are not attributed, instantiate nn.Embedding for them. Only used in first layer
        if isinstance(in_channels_dict, dict):
            non_attr_node_types = (num_nodes_dict.keys() - in_channels_dict.keys())
        else:
            non_attr_node_types = []
        if len(non_attr_node_types) > 0:
            if cpu_embeddings:
                print("Embedding.device = 'cpu'")
                self.embeddings = {node_type: nn.Embedding(num_embeddings=self.num_nodes_dict[node_type],
                                                           embedding_dim=embedding_dim,
                                                           sparse=True).cpu() for node_type in non_attr_node_types}
            else:
                print("Embedding.device = 'gpu'")
                self.embeddings = nn.ModuleDict(
                    {node_type: nn.Embedding(num_embeddings=self.num_nodes_dict[node_type],
                                             embedding_dim=embedding_dim,
                                             sparse=False) for node_type in non_attr_node_types})
        else:
            self.embeddings = None

    def forward(self, node_feats: dict, edge_index_dict: dict, global_node_idx: dict, save_betas=False):
        """
        This
        :param node_feats: Dict of <node_type>:<tensor size (batch_size, in_channels)>. If nodes are not attributed, then pass an empty dict.
        :param global_node_idx: Dict of <node_type>:<int tensor size (batch_size,)>
        :param edge_index_dict: Dict of <metapath>:<tensor size (2, num_edge_index)>
        :param save_betas: whether to save _beta values for batch
        :return embedding_output, proximity_loss, edge_pred_dict:
        """
        proximity_loss = torch.tensor(0.0, device=self.device) if self.use_proximity else None

        h_dict = {}
        for ntype in self.node_types:
            if ntype in node_feats:
                h_dict[ntype] = self.feature_projection[ntype](node_feats[ntype])
            else:
                h_dict[ntype] = self.embeddings[ntype].weight[global_node_idx[ntype]].to(self.device)

        h_layers = {node_type: [] for node_type in global_node_idx}
        for t in range(self.t_order):
            if t == 0:
                h_dict, t_loss, edge_pred_dict = self.layers[t].forward(x_l=h_dict, x_r=h_dict,
                                                                        edge_index_dict=edge_index_dict,
                                                                        global_node_idx=global_node_idx,
                                                                        save_betas=save_betas)
                next_edge_index_dict = edge_index_dict
            else:
                next_edge_index_dict = LATTE.join_edge_indexes(next_edge_index_dict, edge_index_dict, global_node_idx,
                                                               edge_sampling=self.edge_sampling)
                h_dict, t_loss, _ = self.layers[t].forward(x_l=h_dict, x_r=h_dict,
                                                           edge_index_dict=next_edge_index_dict,
                                                           global_node_idx=global_node_idx,
                                                           save_betas=save_betas)

            for node_type in global_node_idx:
                h_layers[node_type].append(h_dict[node_type])

            if self.use_proximity:
                proximity_loss += t_loss

        if self.layer_pooling == "last" or self.t_order == 1:
            out = h_dict

        elif self.layer_pooling == "max":
            out = {node_type: torch.stack(h_list, dim=1) for node_type, h_list in h_layers.items() \
                   if len(h_list) > 0}
            out = {ntype: h_s.max(1).values for ntype, h_s in out.items()}

        elif self.layer_pooling == "mean":
            out = {node_type: torch.stack(h_list, dim=1) for node_type, h_list in h_layers.items() \
                   if len(h_list) > 0}
            out = {ntype: torch.mean(h_s, dim=1) for ntype, h_s in out.items()}

        elif self.layer_pooling == "concat":
            out = {node_type: torch.cat(h_list, dim=1) for node_type, h_list in h_layers.items() \
                   if len(h_list) > 0}
        else:
            raise Exception("`layer_pooling` should be either ['last', 'max', 'mean', 'concat']")

        return out, proximity_loss, edge_pred_dict

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

        elif isinstance(edge_index_tup, torch.Tensor) and edge_index_tup.size(1) > 0:
            edge_index = edge_index_tup
            edge_values = torch.ones(edge_index_tup.size(1), dtype=torch.float64, device=edge_index_tup.device)
        else:
            return None, None

        if edge_values.dtype != torch.float:
            edge_values = edge_values.to(torch.float)

        return edge_index, edge_values

    @staticmethod
    def join_edge_indexes(edge_index_dict_A, edge_index_dict_B, global_node_idx, edge_sampling=False):
        output_edge_index = {}
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
                    new_edge_index, new_values = adamic_adar(indexA=edge_index_a, valueA=values_a,
                                                             indexB=edge_index_b, valueB=values_b,
                                                             m=global_node_idx[metapath_a[0]].size(0),
                                                             k=global_node_idx[metapath_b[0]].size(0),
                                                             n=global_node_idx[metapath_b[-1]].size(0),
                                                             coalesced=True,
                                                             sampling=edge_sampling
                                                             )
                    # print(new_metapath, new_edge_index.shape)
                    if new_edge_index.size(1) == 0: continue
                    output_edge_index[new_metapath] = (new_edge_index, new_values)

                except Exception as e:
                    print(f"{e} \n {metapath_a}: {edge_index_a.size(1)}, {metapath_b}: {edge_index_b.size(1)}")
                    print("\t", {"m": global_node_idx[metapath_a[0]].size(0),
                                 "k": global_node_idx[metapath_a[-1]].size(0),
                                 "n": global_node_idx[metapath_b[-1]].size(0), })
                    continue

        return output_edge_index

    def get_attn_activation_weights(self, t):
        return dict(zip(self.layers[t].metapaths, self.layers[t].alpha_activation.detach().numpy().tolist()))

    def get_relation_weights(self, t):
        return self.layers[t].get_relation_weights()


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
    if isinstance(metapath, tuple) and "neg" in metapath:
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
