import copy
from argparse import Namespace
from typing import List, Dict, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from fairscale.nn import auto_wrap
from torch import nn as nn, Tensor, ModuleDict
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from moge.model.PyG.utils import join_metapaths, get_edge_index_values, join_edge_indexes, max_num_hops, \
    filter_metapaths
from moge.model.relations import RelationAttention


class LATTEConv(MessagePassing, pl.LightningModule, RelationAttention):
    def __init__(self, input_dim: int, output_dim: int, num_nodes_dict: Dict[str, int], metapaths: List,
                 layer: int = 0, t_order: int = 1,
                 activation: str = "relu", attn_heads=4, attn_activation="LeakyReLU", attn_dropout=0.2,
                 layernorm=False, batchnorm=False, dropout=0.2,
                 edge_threshold=0.0, use_proximity=False, neg_sampling_ratio=1.0, verbose=False) -> None:
        super().__init__(aggr="add", flow="source_to_target", node_dim=0)
        self.layer = layer
        self.t_order = t_order
        self.verbose = verbose
        self.node_types = list(num_nodes_dict.keys())
        self.metapaths = list(metapaths)
        print(f"LATTE {self.layer + 1}, metapaths {len(metapaths)}, max_order {max_num_hops(metapaths)}")
        # pprint({ntype: [m[1::2] for m in self.metapaths if m[-1] == ntype] \
        #         for ntype in {m[-1] for m in self.metapaths}}, width=500)

        self.num_nodes_dict = num_nodes_dict
        self.embedding_dim = output_dim
        self.use_proximity = use_proximity
        self.neg_sampling_ratio = neg_sampling_ratio
        self.attn_heads = attn_heads
        self.attn_dropout = attn_dropout
        self.edge_threshold = edge_threshold

        if activation == "sigmoid":
            self.activation = F.sigmoid
        elif activation == "tanh":
            self.activation = F.tanh
        elif activation == "relu":
            self.activation = F.relu
        else:
            print(f"Embedding activation arg `{activation}` did not match, so uses linear activation.")

        self.linear = nn.ModuleDict(
            {node_type: nn.Linear(input_dim, output_dim, bias=True) \
             for node_type in self.node_types})  # W.shape (F x F)
        # self.linear_r = nn.ModuleDict(
        #     {node_type: nn.Linear(input_dim, output_dim, bias=True) \
        #      for node_type in self.node_types})  # W.shape (F x F}

        self.out_channels = self.embedding_dim // attn_heads
        self.attn = nn.Parameter(torch.rand((len(self.metapaths), attn_heads, self.out_channels * 2)))

        self.rel_attn_l = nn.ParameterDict({
            ntype: nn.Parameter(Tensor(attn_heads, self.out_channels)) \
            for ntype in self.node_types})
        self.rel_attn_r = nn.ParameterDict({
            ntype: nn.Parameter(Tensor(self.num_tail_relations(ntype), attn_heads, self.out_channels)) \
            for ntype in self.node_types})

        self.rel_attn_bias = nn.ParameterDict({
            ntype: nn.Parameter(Tensor(self.num_tail_relations(ntype)).fill_(0.0)) \
            for ntype in self.node_types if self.num_tail_relations(ntype) > 1})
        # self.rel_mha = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=attn_heads, dropout=attn_dropout,
        #                                      batch_first=True)

        if attn_activation == "sharpening":
            self.alpha_activation = nn.Parameter(Tensor(len(self.metapaths)).fill_(1.0))
        elif attn_activation == "PReLU":
            self.alpha_activation = nn.PReLU(init=0.2)
        elif attn_activation == "LeakyReLU":
            self.alpha_activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            print(f"WARNING: alpha_activation `{attn_activation}` did not match, so used linear activation")
            self.alpha_activation = None

        if layernorm:
            self.layernorm = {ntype: nn.LayerNorm(output_dim) for ntype in self.node_types}

        if batchnorm:
            self.batchnorm = {ntype: nn.BatchNorm1d(output_dim) for ntype in self.node_types}

        if dropout:
            self.dropout = nn.Dropout(p=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.2)
        for i, metapath in enumerate(self.metapaths):
            nn.init.xavier_normal_(self.attn[i], gain=gain)

        gain = nn.init.calculate_gain('relu')
        for node_type in self.linear:
            nn.init.xavier_normal_(self.linear[node_type].weight, gain=gain)
        # for node_type in self.linear_r:
        #     nn.init.xavier_normal_(self.linear_r[node_type].weight, gain=gain)

        gain = nn.init.calculate_gain('leaky_relu', 0.2)
        if hasattr(self, "rel_attn_l"):
            for ntype, rel_attn in self.rel_attn_l.items():
                nn.init.xavier_normal_(rel_attn, gain=gain)
        if hasattr(self, "rel_attn_r"):
            for ntype, rel_attn in self.rel_attn_r.items():
                nn.init.xavier_normal_(rel_attn, gain=gain)

    # def get_beta_weights(self, query: Tensor, key: Tensor, ntype: str) -> Tensor:
    #     beta_l = (query * self.rel_attn_l[ntype]).sum(dim=-1)
    #     beta_r = (key * self.rel_attn_r[ntype]).sum(dim=-1)
    #
    #     beta = beta_l[:, None, :] + beta_r
    #     beta = F.leaky_relu(beta, negative_slope=0.2)
    #     beta = F.softmax(beta, dim=1)
    #     # beta = F.dropout(beta, p=self.attn_dropout, training=self.training)
    #     return beta

    def get_beta_weights(self, query: Tensor, key: Tensor, ntype: str) -> Tensor:
        beta_l = F.relu(query * self.rel_attn_l[ntype], )
        beta_r = F.relu(key * self.rel_attn_r[ntype], )

        beta = (beta_l[:, None, :, :] * beta_r).sum(-1) + self.rel_attn_bias[ntype][None, :, None]
        beta = F.softmax(beta, dim=1)
        # beta = torch.relu(beta / beta.sum(1, keepdim=True))
        # beta = F.dropout(beta, p=self.attn_dropout, training=self.training)
        return beta

    # def get_beta_weights(self, query: Tensor, key: Tensor, ntype: str):
    #     """
    #     Multihead Attention to find betas
    #     """
    #     key = key.view(query.size(0), self.num_tail_relations(ntype), -1)
    #     _, betas = self.rel_mha.forward(key, key=key, value=key, average_attn_weights=False)
    #
    #     betas = betas.mean(2).permute([0,2,1])
    #     return betas

    def forward(self, feats: Dict[str, Tensor],
                edge_index_dict: Dict[Tuple[str, str, str], Union[Tensor, Tuple[Tensor, Tensor]]],
                global_node_index: Dict[str, Tensor],
                sizes: Dict[str, int],
                edge_pred_dict: Dict[Tuple[str, str, str], Union[Tensor, Tuple[Tensor, Tensor]]],
                save_betas=False, empty_gpu_device=None, verbose=False) -> \
            Tuple[Dict[str, Tensor], Dict[Tuple[str, str, str], Tensor]]:
        """
        Args:
            feats: a dict of node attributes indexed ntype
            global_node_index: A dict of index values indexed by ntype in this mini-batch sampling
            edge_index_dict: Sparse adjacency matrices for each metapath relation. A dict of edge_index indexed by metapath
            sizes: Dict of ntype and number of nodes in `edge_index_dict`
            edge_pred_dict: Higher order edge_index_dict calculated from the previous LATTE layer
        Returns:
             output_emb, edge_attn_scores
        """
        l_dict = self.projection(feats, linears=self.linear)
        r_dict = l_dict
        # r_dict = self.projection(feats, linears=self.linear_r)

        print("\nLayer", self.layer + 1, ) if verbose else None

        # For each metapath in a node_type, use GAT message passing to aggregate h_j neighbors
        h_out = {}
        betas = {}
        edge_pred_dicts = {}
        for ntype in global_node_index:
            if global_node_index[ntype].size(0) == 0 or self.num_tail_relations(ntype) <= 1: continue
            embedding, edge_pred_dict = self.aggregate_relations(
                ntype=ntype, l_dict=l_dict, r_dict=r_dict,
                edge_index_dict=edge_index_dict, edge_pred_dict=edge_pred_dict, sizes=sizes)
            edge_pred_dicts.update(edge_pred_dict)

            embedding[:, -1] = l_dict[ntype]

            # Soft-select the relation-specific embeddings by a weighted average with beta[node_type]
            betas[ntype] = self.get_beta_weights(query=r_dict[ntype], key=embedding, ntype=ntype)

            if verbose:
                print("  >", ntype, global_node_index[ntype].shape, )
                for i, (metapath, beta_mean, beta_std) in enumerate(
                        zip(self.get_tail_relations(ntype) + [ntype],
                            betas[ntype].mean(-1).mean(0),
                            betas[ntype].mean(-1).std(0))):
                    print(f"   - {'.'.join(metapath[1::2]) if isinstance(metapath, tuple) else metapath}, "
                          f"\tedge_index: {edge_pred_dict[metapath][0].size(1) if metapath in edge_pred_dict else 0}, "
                          f"\tbeta: {beta_mean.item():.2f} ± {beta_std.item():.2f}, "
                          f"\tnorm: {torch.norm(embedding[:, i]).item():.2f}")

            embedding = embedding * betas[ntype].unsqueeze(-1)
            embedding = embedding.sum(1).view(embedding.size(0), self.embedding_dim)

            if hasattr(self, "activation"):
                embedding = self.activation(embedding)

            if hasattr(self, "layernorm"):
                embedding = self.layernorm[ntype](embedding)
            elif hasattr(self, "batchnorm"):
                embedding = self.batchnorm[ntype](embedding)

            if hasattr(self, "dropout"):
                embedding = self.dropout(embedding)

            h_out[ntype] = embedding

        if save_betas:
            self.save_relation_weights(betas={ntype: betas[ntype].mean(-1) for ntype in betas},
                                       # mean on attn_heads dim
                                       global_node_index=global_node_index)

        return h_out, edge_pred_dicts

    def projection(self, feats: Dict[str, Tensor], linears: ModuleDict):
        h_dict = {ntype: linears[ntype].forward(x).relu_() for ntype, x in feats.items()}

        h_dict = {ntype: h_dict[ntype].view(feats[ntype].size(0), self.attn_heads, self.out_channels) \
                  for ntype in h_dict}

        return h_dict

    def aggregate_relations(self, ntype: str,
                            l_dict: Dict[str, Tensor],
                            r_dict: Dict[str, Tensor],
                            edge_index_dict: Dict[Tuple[str, str, str], Tensor],
                            edge_pred_dict: Dict[Tuple[str, str, str], Union[Tensor, Tuple[Tensor, Tensor]]],
                            sizes: Dict[str, int]):
        # Initialize embeddings, size: (num_nodes, num_relations, embedding_dim)
        emb_relations = torch.zeros(
            size=(sizes[ntype],
                  self.num_tail_relations(ntype),
                  self.attn_heads,
                  self.out_channels)).type_as(self.attn)
        relations = self.get_tail_relations(ntype)

        if edge_pred_dict is None:
            edge_pred_dict = copy.copy(edge_index_dict)
        elif len(edge_pred_dict) < len(edge_index_dict):
            edge_pred_dict.update(edge_index_dict)

        # First order
        for metapath in self.get_tail_relations(ntype, order=1):
            if metapath not in edge_index_dict or edge_index_dict[metapath] is None: continue
            head, tail = metapath[0], metapath[-1]
            num_node_head, num_node_tail = sizes[head], sizes[tail]

            edge_index, values = get_edge_index_values(edge_index_dict[metapath], filter_edge=False)
            if edge_index is None or edge_index.size(1) == 0: continue

            # Propapate flows from target nodes to source nodes
            out = self.propagate(
                edge_index=edge_index,
                x=(l_dict[head], r_dict[tail]),
                size=(num_node_head, num_node_tail),
                metapath_idx=self.metapaths.index(metapath),
                values=None)

            emb_relations[:, relations.index(metapath)] = out
            edge_pred_dict[metapath] = (edge_index, self._alpha)
            self._alpha = None

        remaining_orders = list(range(2, min(self.layer + 1, self.t_order) + 1))
        higher_relations = self.get_tail_relations(ntype, order=remaining_orders)

        # Create high-order edge index for next layer (but may not be used for aggregation)
        if len(edge_pred_dict) < len(higher_relations):
            higher_order_edge_index = join_edge_indexes(edge_index_dict_A=edge_pred_dict,
                                                        edge_index_dict_B=edge_index_dict,
                                                        sizes=sizes,
                                                        filter_metapaths=higher_relations,
                                                        edge_threshold=None,
                                                        # device=self.empty_gpu_device
                                                        )
        else:
            higher_order_edge_index = edge_pred_dict

        # Aggregate higher order relations
        for metapath in higher_relations:
            if metapath not in higher_order_edge_index or higher_order_edge_index[metapath] == None: continue

            edge_index, values = get_edge_index_values(higher_order_edge_index[metapath], filter_edge=False)
            if edge_index is None or edge_index.size(1) == 0: continue

            head, tail = metapath[0], metapath[-1]
            head_size_in, tail_size_out = sizes[head], sizes[tail]

            # Select the right t-order context node presentations based on the order of the metapath
            # Propapate flows from higher order source nodes to target nodes
            out = self.propagate(
                edge_index=edge_index,
                x=(l_dict[head], r_dict[tail]),
                size=(head_size_in, tail_size_out),
                metapath_idx=self.metapaths.index(metapath),
                values=None)
            emb_relations[:, relations.index(metapath)] = out

            edge_pred_dict[metapath] = (edge_index, self._alpha)
            self._alpha = None

        return emb_relations, edge_pred_dict

    def message(self, x_j, x_i, index, ptr, size_i, metapath_idx, values=None):
        if values is None:
            x = torch.cat([x_i, x_j], dim=2)

            # Sharpening alpha values
            if isinstance(self.alpha_activation, Tensor):
                x = self.alpha_activation[metapath_idx] * F.leaky_relu(x, negative_slope=0.2)
            # Non-linear activation
            elif isinstance(self.alpha_activation, nn.Module):
                x = self.alpha_activation(x)

            alpha = (x * self.attn[metapath_idx]).sum(dim=-1)
            alpha = softmax(alpha, index=index, ptr=ptr, num_nodes=size_i)
        else:
            if values.dim() == 1:
                values = values.unsqueeze(-1)
            alpha = values
            # alpha = softmax(alpha, index=index, ptr=ptr, num_nodes=size_i)

        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.attn_dropout, training=self.training)

        return x_j * alpha.unsqueeze(-1)

    def attn_activation(self, alpha, metapath_id):
        if isinstance(self.alpha_activation, Tensor):
            return self.alpha_activation[metapath_id] * alpha
        elif isinstance(self.alpha_activation, nn.Module):
            return self.alpha_activation(alpha)
        else:
            return alpha


class LATTE(nn.Module):
    def __init__(self, n_layers: int, t_order: int,
                 embedding_dim: int, num_nodes_dict: Dict[str, int],
                 metapaths: List[Tuple[str, str, str]], layer_pooling: str = None,
                 activation: str = "relu",
                 attn_heads: int = 1, attn_activation="sharpening", attn_dropout: float = 0.5,
                 use_proximity=True, neg_sampling_ratio=2.0, edge_sampling=True,
                 hparams: Namespace = None):
        super().__init__()
        self.metapaths = metapaths
        self.node_types = list(num_nodes_dict.keys())
        self.head_node_type = hparams.head_node_type

        self.embedding_dim = embedding_dim
        self.t_order = t_order
        self.n_layers = n_layers

        self.neg_sampling_ratio = neg_sampling_ratio
        self.edge_sampling = edge_sampling
        self.use_proximity = use_proximity
        self.layer_pooling = layer_pooling

        layer_t_orders = {
            l: list(range(1, t_order - (n_layers - (l + 1)) + 1)) \
                if (t_order - (n_layers - (l + 1))) > 0 \
                else [1] \
            for l in reversed(range(n_layers))}
        # layer_t_orders = {
        #     l: list(range(1, t_order + 1)) \
        #     for l in range(n_layers)}

        higher_order_metapaths = copy.deepcopy(metapaths)  # Initialize another set of meapaths

        layers = []
        for l in range(n_layers):
            is_last_layer = l + 1 == n_layers

            l_layer_metapaths = filter_metapaths(
                metapaths=metapaths + higher_order_metapaths,
                order=layer_t_orders[l],  # Select only up to t-order
                # Skip higher-order relations that doesn't have the head node type, since it's the last output layer.
                tail_type=[self.head_node_type, "GO_term"] if is_last_layer else None)

            layer = LATTEConv(input_dim=embedding_dim,
                              output_dim=embedding_dim,
                              num_nodes_dict=num_nodes_dict,
                              metapaths=l_layer_metapaths,
                              layer=l,
                              t_order=self.t_order,
                              activation=activation,
                              layernorm=hparams.layernorm,
                              batchnorm=hparams.batchnorm,
                              dropout=hparams.dropout if "dropout" in hparams else 0.0,
                              attn_heads=attn_heads,
                              attn_activation=attn_activation,
                              attn_dropout=attn_dropout,
                              edge_threshold=hparams.edge_threshold if "edge_threshold" in hparams else 0.0,
                              use_proximity=use_proximity,
                              neg_sampling_ratio=neg_sampling_ratio,
                              verbose=hparams.verbose if "verbose" in hparams else False)
            if l + 1 < n_layers and layer_t_orders[l + 1] > layer_t_orders[l]:
                higher_order_metapaths = join_metapaths(l_layer_metapaths, metapaths)

            layers.append(layer)

        self.layers: List[LATTEConv] = nn.ModuleList(layers)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')

    def configure_sharded_model(self):
        # modules are sharded across processes
        # as soon as they are wrapped with ``wrap`` or ``auto_wrap``.
        # During the forward/backward passes, weights get synced across processes
        # and de-allocated once computation is complete, saving memory.

        # Wraps the layer in a Fully Sharded Wrapper automatically
        self.layers = auto_wrap(self.layers)

    def forward(self, h_dict: Dict[str, Tensor],
                edge_index_dict: Dict[Tuple[str, str, str], Tensor],
                global_node_index: Dict[str, Tensor],
                sizes: Dict[str, int],
                **kwargs):
        """
        Args:
            h_dict: Dict of <ntype>:<tensor size (batch_size, in_channels)>. If nodes are not attributed, then pass an empty dict.
            global_node_index: Dict of <ntype>:<int tensor size (batch_size,)>
            edge_index_dict: Dict of <metapath>:<tensor size (2, num_edge_index)>
            save_betas: whether to save _beta values for batch
        Returns:
            embedding_output
        """
        h_layers = {ntype: [] for ntype in global_node_index}
        edge_pred_dict = None
        for l in range(self.n_layers):
            h_dict, edge_pred_dict = self.layers[l].forward(feats=h_dict, edge_index_dict=edge_index_dict,
                                                            global_node_index=global_node_index,
                                                            sizes=sizes,
                                                            edge_pred_dict=edge_pred_dict,
                                                            **kwargs)

            for ntype in h_dict:
                h_layers[ntype].append(h_dict[ntype])

        if self.layer_pooling is None or self.layer_pooling in ["last"] or self.n_layers == 1:
            out = h_dict

        elif self.layer_pooling == "concat":
            out = {node_type: torch.cat(h_list, dim=1) for node_type, h_list in h_layers.items() \
                   if len(h_list)}
        else:
            raise Exception("`layer_pooling` should be either ['last', 'max', 'mean', 'concat']")

        return out

    def __getitem__(self, item) -> LATTEConv:
        return self.layers[item]
