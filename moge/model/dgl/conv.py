import dgl
import torch
from dgl import function as fn
from dgl.nn import pytorch as dglnn
from dgl.nn.pytorch import GATConv
from torch import nn
from torch.nn import functional as F
from transformers import BertForSequenceClassification


class StochasticTwoLayerRGCN(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(in_feat, hidden_feat, num_heads=4)
            for rel in rel_names
        })
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(hidden_feat, out_feat, num_heads=4)
            for rel in rel_names
        })

    def forward(self, blocks, x):
        x = self.conv1(blocks[0], x)
        x = self.conv2(blocks[1], x)
        return x


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * z).sum(1)  # (N, D * K)


class HANLayer(torch.nn.Module):
    """
    HAN layer.
    Arguments
    ---------
    num_metapath : number of metapath based sub-graph
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability
    Inputs
    ------
    g : DGLHeteroGraph
        The heterogeneous graph
    h : tensor
        Input features
    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(self, num_metapath, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(num_metapath):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu,
                                           allow_zero_in_degree=True))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.num_metapath = num_metapath

    def forward(self, block_list, h_list):
        semantic_embeddings = []

        for i, block in enumerate(block_list):
            semantic_embeddings.append(self.gat_layers[i](block, h_list[i]).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)

class HAN(nn.Module):
    def __init__(self, num_metapath, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(num_metapath, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(num_metapath, hidden_size * num_heads[l - 1],
                                        hidden_size, num_heads[l], dropout))
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)

        return self.predict(h)


class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes, agg='stack'):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation
        self.weight = nn.ModuleDict({
            name: nn.Linear(in_size, out_size) for name in etypes
        })
        self.agg = agg

    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            # Compute W_r * h
            Wh = self.weight[etype](feat_dict[srctype])
            # Save it in graph for message passing
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            # Specify per-relation message passing functions: (message_func, reduce_func).
            # Note that the results are saved to the same destination feature 'h', which
            # hints the type wise reducer for aggregation.
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))

        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        G.multi_update_all(funcs, self.agg)
        # return the updated node feature dictionary
        return {ntype: G.nodes[ntype].data['h'] for ntype in G.ntypes}


class HeteroRGCN(nn.Module):
    def __init__(self, G: dgl.DGLHeteroGraph, in_size, hidden_size, out_size,
                 encoder: BertForSequenceClassification = None):
        super(HeteroRGCN, self).__init__()

        # Use trainable node embeddings as featureless inputs.
        if encoder is not None and "input_ids" in G.node_attr_schemes():
            self.encoder = encoder
        else:
            self.embeddings = nn.ParameterDict(
                {ntype: nn.Embedding(G.num_nodes(ntype), embedding_dim=in_size) for ntype in G.ntypes})

        # create layers
        self.layer1 = HeteroRGCNLayer(in_size, hidden_size, G.etypes, agg="mean")
        self.layer2 = HeteroRGCNLayer(hidden_size, out_size, G.etypes, agg="mean")

    def forward(self, G: dgl.DGLHeteroGraph):
        if hasattr(self, "embeddings"):
            feats = {ntype: self.embeddings[ntype].weights for ntype in G.ntypes}
        elif hasattr(self, "encoder"):
            feats = {}
            for ntype in G.ntypes:
                out = self.encoder.forward(G.nodes[ntype].data["input_ids"],
                                           G.nodes[ntype].data["attention_mask"],
                                           G.nodes[ntype].data["token_type_ids"])
                feats[ntype] = out.logits

        h_dict = self.layer1(G, feats)
        h_dict = {k: F.relu(h) for k, h in h_dict.items()}
        h_dict = self.layer2(G, h_dict)

        return h_dict
