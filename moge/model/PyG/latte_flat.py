import copy
import traceback
from argparse import Namespace
from typing import List, Dict, Tuple, Union, Set

import torch
import torch.nn.functional as F
from fairscale.nn import auto_wrap
from torch import nn as nn, Tensor, ModuleDict
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from moge.model.PyG.relations import RelationAttention, RelationMultiLayerAgg
from moge.model.PyG.utils import join_metapaths, get_edge_index_values, join_edge_indexes, max_num_hops, \
    filter_metapaths


class LATTEConv(MessagePassing, RelationAttention):
    def __init__(self, input_dim: int, output_dim: int, num_nodes_dict: Dict[str, int], metapaths: List,
                 layer: int = 0, t_order: int = 1,
                 activation: str = "relu", attn_heads=4, attn_activation="LeakyReLU", attn_dropout=0.2,
                 layernorm=False, batchnorm=False, dropout=0.2,
                 edge_threshold=0.0, n_layers=None, verbose=False) -> None:
        """
        Create a LATTEConv layer for a given set of metapaths and node types.
        Args:
            input_dim:
            output_dim:
            num_nodes_dict:
            metapaths:
            layer:
            t_order:
            activation:
            attn_heads:
            attn_activation:
            attn_dropout:
            layernorm:
            batchnorm:
            dropout:
            edge_threshold:
            n_layers:
            verbose:
        """
        super().__init__(aggr="add", flow="source_to_target", node_dim=0)
        self.verbose = verbose

        self.layer = layer
        self.n_layers = n_layers
        self.is_last_layer = (self.layer + 1) == self.n_layers
        self.t_order = t_order

        self.node_types = list(num_nodes_dict.keys())
        self.metapaths = list(metapaths)
        print(f"LATTE {self.layer + 1}, metapaths {len(metapaths)}, max_order {max_num_hops(metapaths)}")

        self.num_nodes_dict = num_nodes_dict
        self.embedding_dim = output_dim
        self.attn_heads = attn_heads
        self.attn_dropout = attn_dropout
        self.edge_threshold = edge_threshold

        if activation == "sigmoid":
            self.activation = F.sigmoid
        elif activation == "tanh":
            self.activation = F.tanh
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "elu":
            self.activation = F.elu
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
            ntype: nn.Parameter(Tensor(self.num_tail_relations(ntype), attn_heads, self.out_channels)) \
            for ntype in self.node_types})
        self.rel_attn_r = nn.ParameterDict({
            ntype: nn.Parameter(Tensor(self.num_tail_relations(ntype), attn_heads, self.out_channels)) \
            for ntype in self.node_types})
        self.rel_attn_w = nn.ParameterDict({
            ntype: nn.Parameter(
                Tensor(self.num_tail_relations(ntype, include_self=False), attn_heads, self.out_channels)) \
            for ntype in self.node_types})
        self.rel_attn_bias = nn.ParameterDict({
            ntype: nn.Parameter(Tensor(self.num_tail_relations(ntype)).fill_(0.0)) \
            for ntype in self.node_types if self.num_tail_relations(ntype) > 1})

        # self.relation_conv: Dict[str, MetapathGATConv] = nn.ModuleDict({
        #     ntype: MetapathGATConv(output_dim, metapaths=self.get_tail_relations(ntype), n_layers=1,
        #                            attn_heads=attn_heads, attn_dropout=attn_dropout) \
        #     for ntype in self.node_types})

        if attn_activation == "PReLU":
            self.alpha_activation = nn.PReLU(init=0.2)
        elif attn_activation == "LeakyReLU":
            self.alpha_activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            print(f"WARNING: alpha_activation `{attn_activation}` did not match, so used linear activation")
            self.alpha_activation = None

        if layernorm:
            self.layernorm = nn.LayerNorm(output_dim)
        if batchnorm:
            self.batchnorm = nn.BatchNorm1d(output_dim)

        if dropout:
            self.dropout = nn.Dropout(p=dropout)

        self.reset_parameters()

    def extra_repr(self) -> str:
        return 'linear_l={}, linear_r={}, attn={}, bias={}'.format(
            self.linear, self.linear_r if hasattr(self, 'linear_r') else None, self.attn,
            self.bias is not None)

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.2)
        for i, metapath in enumerate(self.metapaths):
            nn.init.xavier_normal_(self.attn[i], gain=gain)

        gain = nn.init.calculate_gain('relu')
        for node_type in self.linear:
            nn.init.xavier_normal_(self.linear[node_type].weight, gain=gain)
            if hasattr(self, 'linear_r'):
                nn.init.xavier_normal_(self.linear_r[node_type].weights, gain=gain)

        gain = nn.init.calculate_gain('leaky_relu', 0.2)
        if hasattr(self, "rel_attn_l"):
            for ntype, rel_attn in self.rel_attn_l.items():
                nn.init.xavier_normal_(rel_attn, gain=gain)
        if hasattr(self, "rel_attn_r"):
            for ntype, rel_attn in self.rel_attn_r.items():
                nn.init.xavier_normal_(rel_attn, gain=gain)
        if hasattr(self, "rel_attn_w"):
            for ntype, rel_attn in self.rel_attn_w.items():
                nn.init.xavier_normal_(rel_attn, gain=1.0)

    # def get_beta_weights(self, query: Tensor, key: Tensor, ntype: str) -> Tensor:
    #     beta_l = (query * self.rel_attn_l[ntype]).sum(dim=-1)
    #     beta_r = (key * self.rel_attn_r[ntype]).sum(dim=-1)
    #
    #     beta = beta_l[:, None, :] + beta_r
    #     beta = F.leaky_relu(beta, negative_slope=0.2)
    #     beta = F.softmax(beta, dim=1)
    #     # beta = F.dropout(beta, p=self.attn_dropout, training=self.training)
    #     return beta

    # def get_beta_weights(self, query: Tensor, key: Tensor, ntype: str) -> Tensor:
    #     beta_l = F.relu(query * self.rel_attn_l[ntype])
    #     beta_r = F.relu(key * self.rel_attn_r[ntype])
    #
    #     beta = (beta_l[:, None, :, :] * beta_r).sum(-1) + self.rel_attn_bias[ntype][None, :, None]
    #     beta = F.softmax(beta, dim=1)
    #     # beta = F.dropout(beta, p=self.attn_dropout, training=self.training)
    #     return beta

    def get_beta_weights(self, query: Tensor, key: Tensor, ntype: str) -> Tensor:
        x_l = query[:, None, :] * self.rel_attn_l[ntype]
        x_r = key * self.rel_attn_r[ntype]

        bias = self.rel_attn_bias[ntype][None, :, None]  # / torch.sqrt(self.embedding_dim)
        beta = F.leaky_relu((x_l + x_r).sum(-1), 0.2) + bias
        # print(tensor_sizes(beta=beta, x_l=x_l, x_r=x_r, rel_attn_bias=self.rel_attn_bias[ntype]))
        beta = F.softmax(beta.transpose(2, 1), dim=2).transpose(2, 1)
        beta = F.dropout(beta, p=self.attn_dropout, training=self.training)
        return beta

    def projection(self, feats: Dict[str, Tensor], linears: ModuleDict, subset: Set[str] = None):
        h_dict = {ntype: linears[ntype].forward(x) \
                  for ntype, x in feats.items() if not subset or ntype in subset}

        h_dict = {ntype: h.view(feats[ntype].size(0), self.attn_heads, self.out_channels) \
                  for ntype, h in h_dict.items()}

        return h_dict

    def forward(self, feats: Dict[str, Tensor],
                edge_index_dict: Dict[Tuple[str, str, str], Union[Tensor, Tuple[Tensor, Tensor]]],
                global_node_index: Dict[str, Tensor],
                edge_pred_dict: Dict[Tuple[str, str, str], Union[Tensor, Tuple[Tensor, Tensor]]],
                batch_sizes: Dict[str, int],
                save_betas=False, verbose=False) -> \
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
        l_dict = self.projection(feats, linears=self.linear, subset=self.get_src_ntypes())
        # r_dict = self.projection(feats, linears=self.linear_r, subset=self.get_dst_ntypes())
        r_dict = l_dict

        print("\nLayer", self.layer + 1, ) if verbose else None

        # For each metapath in a node_type, use GAT message passing to aggregate h_j neighbors
        h_out = {}
        betas = {}
        edge_pred_dicts = {}
        for ntype in global_node_index:
            if global_node_index[ntype].size(0) == 0 or self.num_tail_relations(ntype) <= 1: continue

            h_out[ntype], edge_pred_dict = self.aggregate_relations(
                ntype=ntype, l_dict=l_dict, r_dict=r_dict,
                edge_index_dict=edge_index_dict,
                edge_pred_dict=edge_pred_dict,
                global_node_index=global_node_index,
                verbose=verbose)
            edge_pred_dicts.update(edge_pred_dict)

            h_out[ntype][:, -1] = l_dict[ntype]

            if verbose:
                rel_embedding = h_out[ntype].detach().clone()

            # GATRelAttn
            # h_out[ntype] = h_out[ntype].view(h_out[ntype].size(0), self.num_tail_relations(ntype), self.embedding_dim)
            # h_out[ntype], betas[ntype] = self.relation_conv[ntype].forward(h_out[ntype])

            # Soft-select the relation-specific embeddings by a weighted average with beta[node_type]
            betas[ntype] = self.get_beta_weights(query=r_dict[ntype], key=h_out[ntype], ntype=ntype)

            if verbose:
                print("  >", ntype, h_out[ntype].shape, betas[ntype].shape)
                for i, (metapath, beta_mean, beta_std) in enumerate(zip(self.get_tail_relations(ntype) + [ntype],
                                                                        betas[ntype].mean(-1).mean(0),
                                                                        betas[ntype].mean(-1).std(0))):
                    if metapath in edge_pred_dict:
                        edge_index = get_edge_index_values(edge_pred_dict[metapath], drop_edge_value=True)
                        edge_size = edge_index.size(1)
                    elif metapath != ntype:
                        continue
                    else:
                        edge_size = None

                    rel_embedding = h_out[ntype] if h_out[ntype].dim() >= 3 else rel_embedding
                    print(f"   - {'.'.join(metapath[1::2]) if isinstance(metapath, tuple) else metapath}, "
                          f"\tedge_index: {edge_size}, "
                          f"\tbeta: {beta_mean.item():.2f} ± {beta_std.item():.2f}, "
                          f"\tnorm: {torch.norm(rel_embedding[:, i].view(h_out[ntype].size(0), -1), dim=1).mean(dim=0).item() :.2f}")

            if hasattr(self, 'rel_attn_w'):
                h_out[ntype] = h_out[ntype] * torch.cat([self.rel_attn_w[ntype],
                                                         torch.ones_like(self.rel_attn_w[ntype][[0]])], dim=0)
            h_out[ntype] = h_out[ntype] * betas[ntype].unsqueeze(-1)
            h_out[ntype] = h_out[ntype].sum(1).view(h_out[ntype].size(0), self.embedding_dim)

            if hasattr(self, "activation"):
                h_out[ntype] = self.activation(h_out[ntype])

            if hasattr(self, "dropout"):
                h_out[ntype] = self.dropout(h_out[ntype])

            if hasattr(self, "layernorm"):
                h_out[ntype] = self.layernorm(h_out[ntype])
            if hasattr(self, "batchnorm"):
                h_out[ntype] = self.batchnorm(h_out[ntype])

            if verbose:
                print(f"   -> {self.activation.__name__ if hasattr(self, 'activation') else ''} "
                      f"{'batchnorm' if hasattr(self, 'batchnorm') else ''} "
                      f"{'batchnorm' if hasattr(self, 'batchnorm') else ''} "
                      f"{'layernorm' if hasattr(self, 'layernorm') else ''}: "
                      f"{torch.norm(h_out[ntype], dim=1).mean().item():.2f}")

        if save_betas:
            try:
                self.update_relation_attn(betas={ntype: betas[ntype].mean(-1) for ntype in betas},
                                          global_node_index=global_node_index,
                                          batch_sizes=batch_sizes if self.is_last_layer else None)
                if save_betas >= 2:
                    self.update_edge_attn(edge_index_dict=edge_pred_dict,
                                          global_node_index=global_node_index,
                                          batch_sizes=batch_sizes if self.is_last_layer else None,
                                          save_count_only=save_betas == 2)
            except Exception as e:
                traceback.print_exc()

        return h_out, edge_pred_dicts

    def aggregate_relations(self, ntype: str,
                            l_dict: Dict[str, Tensor],
                            r_dict: Dict[str, Tensor],
                            edge_index_dict: Dict[Tuple[str, str, str], Tensor],
                            edge_pred_dict: Dict[Tuple[str, str, str], Union[Tensor, Tuple[Tensor, Tensor]]],
                            global_node_index: Dict[str, Tensor],
                            verbose=False) \
            -> Tuple[Tensor, Dict[Tuple, Tuple[Tensor, Tensor]]]:
        """

        Args:
            ntype ():
            l_dict ():
            r_dict ():
            edge_index_dict ():
            edge_pred_dict ():
            global_node_index ():
            verbose ():

        Returns:
            emb_relations:
            edge_pred_dict:
        """
        sizes = {ntype: nids.numel() for ntype, nids in global_node_index.items()}

        # Initialize embeddings, size: (num_nodes, num_relations, embedding_dim)
        emb_relations = torch.zeros(size=(sizes[ntype],
                                          self.num_tail_relations(ntype, include_self=True),
                                          self.attn_heads,
                                          self.out_channels)).type_as(self.attn)
        ntype_metapaths = self.get_tail_relations(ntype)

        if edge_pred_dict is None:
            edge_pred_dict = copy.copy(edge_index_dict)
        elif set(edge_index_dict).difference(edge_pred_dict):
            edge_pred_dict.update(edge_index_dict)

        # First order
        for metapath in self.get_tail_relations(ntype, order=1):
            head, tail = metapath[0], metapath[-1]
            ntype_metapath_idx = ntype_metapaths.index(metapath)

            if metapath not in edge_index_dict or edge_index_dict[metapath] is None or \
                    head not in sizes or tail not in sizes: continue
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

            emb_relations[:, ntype_metapath_idx] = out
            edge_pred_dict[metapath] = (edge_index, self._alpha)
            self._alpha = None

        remaining_orders = list(range(2, min(self.layer + 1, self.t_order) + 1))
        higher_relations = self.get_tail_relations(ntype, order=remaining_orders)

        # Create high-order edge index for next layer (but may not be used for aggregation)
        if set(filter_metapaths(higher_relations, tail_type=ntype)).difference(edge_pred_dict):
            # print(">>", self.layer, ntype, len(set(filter_metapaths(higher_relations, tail_type=ntype)).difference(edge_pred_dict)))
            higher_order_edge_index = join_edge_indexes(edge_index_dict_A=edge_pred_dict,
                                                        edge_index_dict_B=edge_index_dict,
                                                        sizes=sizes,
                                                        filter_metapaths=higher_relations,
                                                        use_edge_values=False,
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
            emb_relations[:, ntype_metapaths.index(metapath)] = out

            edge_pred_dict[metapath] = (edge_index, self._alpha)
            self._alpha = None

        return emb_relations, edge_pred_dict

    def message(self, x_j, x_i, index, ptr, size_i, metapath_idx, values=None):
        if values is None:
            x = torch.cat([x_i, x_j], dim=2)

            # Non-linear activation
            if isinstance(self.alpha_activation, nn.Module):
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

class LATTE(nn.Module, RelationMultiLayerAgg):
    def __init__(self, n_layers: int, t_order: int,
                 embedding_dim: int,
                 num_nodes_dict: Dict[str, int],
                 metapaths: List[Tuple[str, str, str]],
                 layer_pooling: str = None,
                 activation: str = "relu",
                 attn_heads: int = 1,
                 attn_activation="sharpening",
                 attn_dropout: float = 0.5,
                 edge_sampling=True,
                 hparams: Namespace = None):
        """
        Create a LATTE model with multiple layers of GAT message passing on a heterogeneous graph and generating
        higher-order edge types.

        Args:
            n_layers:
            t_order:
            embedding_dim:
            num_nodes_dict:
            metapaths:
            layer_pooling:
            activation:
            attn_heads:
            attn_activation:
            attn_dropout:
            edge_sampling:
            hparams:
        """
        super().__init__()
        self.metapaths = metapaths
        self.node_types = list(num_nodes_dict.keys())
        self.head_node_type = hparams.head_node_type

        self.embedding_dim = embedding_dim
        self.t_order = t_order
        self.n_layers = n_layers

        self.edge_sampling = edge_sampling
        self.layer_pooling = layer_pooling

        higher_order_metapaths = copy.deepcopy(metapaths)  # Initialize another set of meapaths
        if hasattr(hparams, 'pred_ntypes') and hparams.pred_ntypes:
            pred_ntypes = hparams.pred_ntypes
            output_ntypes = [self.head_node_type] + \
                            (pred_ntypes.split(' ') if isinstance(pred_ntypes, str) else pred_ntypes)
        else:
            output_ntypes = [self.head_node_type]

        layers = []
        for l in range(n_layers):
            is_last_layer = l + 1 == n_layers
            layer_t_order = min(l + 1, t_order) if n_layers >= t_order else t_order

            while max_num_hops(higher_order_metapaths) < layer_t_order:
                higher_order_metapaths = join_metapaths(higher_order_metapaths, metapaths, skip_undirected=False)

            l_layer_metapaths = filter_metapaths(metapaths=metapaths + higher_order_metapaths,
                                                 # order=list(range(1, layer_t_order + 1)),
                                                 order=layer_t_order,
                                                 tail_type=output_ntypes if is_last_layer else None,
                                                 filter=hparams.filter_metapaths if 'filter_metapaths' in hparams else None,
                                                 exclude=hparams.exclude_metapaths if 'exclude_metapaths' in hparams else None,
                                                 )

            layer = LATTEConv(input_dim=embedding_dim, output_dim=embedding_dim, num_nodes_dict=num_nodes_dict,
                              metapaths=l_layer_metapaths, layer=l, t_order=self.t_order, activation=activation,
                              layernorm=hparams.layernorm, batchnorm=hparams.batchnorm,
                              dropout=getattr(hparams, 'dropout', 0.0), attn_heads=attn_heads,
                              attn_activation=attn_activation, attn_dropout=attn_dropout,
                              edge_threshold=getattr(hparams, 'edge_threshold', 0.0), n_layers=n_layers,
                              verbose=getattr(hparams, 'verbose', False), )
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
                batch_sizes: Dict[str, int],
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
                                                            edge_pred_dict=edge_pred_dict,
                                                            batch_sizes=batch_sizes,
                                                            **kwargs)

            for ntype in h_dict:
                h_layers[ntype].append(h_dict[ntype])

        if self.layer_pooling is None or self.layer_pooling in ["last"] or self.n_layers == 1:
            h_dict = h_dict

        elif self.layer_pooling == "concat":
            h_dict = {ntype: torch.cat(h_list, dim=1) for ntype, h_list in h_layers.items() \
                      if len(h_list)}
        else:
            raise Exception("`layer_pooling` should be either ['last', 'max', 'mean', 'concat']")

        return h_dict

    def __getitem__(self, item) -> LATTEConv:
        return self.layers[item]
