import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from moge.generator import HeteroNetDataset
from moge.module.PyG.latte import LATTE
from moge.module.losses import LinkPredLoss
from ..trainer import LinkPredTrainer


class DistMulti(torch.nn.Module):
    def __init__(self, embedding_dim, metapaths):
        super(DistMulti, self).__init__()
        self.metapaths = metapaths
        self.embedding_dim = embedding_dim

        # self.linears = nn.ModuleDict(
        #     {metapath: nn.Parameter(torch.Tensor((embedding_dim, embedding_dim))) \
        #      for metapath in metapaths}
        # )

        self.relation_embedding = nn.Parameter(torch.zeros(len(metapaths), embedding_dim), requires_grad=True)
        nn.init.uniform_(tensor=self.relation_embedding, a=-1, b=1)

    def forward(self, inputs, embeddings):
        output = {}

        # Single edges
        output["edge_pos"] = self.predict(inputs["edge_pos"], embeddings, mode="single")

        # Sampled head or tail batch
        if "head-batch" in inputs or "tail-batch" in inputs:
            # Head batch
            edge_head_batch, neg_samp_size = self.get_edge_index_from_neg_batch(inputs["edge_pos"],
                                                                                neg_edges=inputs["head-batch"],
                                                                                mode="head")
            output["head-batch"] = self.predict(edge_head_batch, embeddings, mode="head")

            # Tail batch
            edge_tail_batch, neg_samp_size = self.get_edge_index_from_neg_batch(inputs["edge_pos"],
                                                                                neg_edges=inputs["tail-batch"],
                                                                                mode="tail")
            output["tail-batch"] = self.predict(edge_tail_batch, embeddings, mode="tail")

        # Single edges
        elif "edge_neg" in inputs:
            output["edge_neg"] = self.predict(inputs["edge_neg"], embeddings, mode="single")
        else:
            raise Exception(f"No negative edges in inputs {inputs.keys()}")

        return output

    def predict(self, edge_index_dict: dict, embeddings: dict, mode: str):
        edge_pred_dict = {}

        for metapath, edge_index in edge_index_dict.items():
            metapath_idx = self.metapaths.index(metapath)
            kernel = self.relation_embedding[metapath_idx]  # (emb_dim)

            if "tail" == mode:
                side_A = (embeddings[metapath[0]] * kernel)[edge_index[0]].unsqueeze(1)  # (n_nodes, 1, emb_dim)
                emb_B = embeddings[metapath[-1]][edge_index[1]].unsqueeze(2)  # (n_nodes, emb_dim, 1)
                # score = side_A * emb_B
                score = torch.bmm(side_A, emb_B)
                score = score.sum(-1)
            else:
                emb_A = embeddings[metapath[0]][edge_index[0]].unsqueeze(1)  # (n_nodes, 1, emb_dim)
                side_B = (kernel * embeddings[metapath[-1]])[edge_index[1]].unsqueeze(2)  # (n_nodes, emb_dim, 1)

                # score = emb_A * side_B
                score = torch.bmm(emb_A, side_B)
                score = score.sum(-1)

            score = score.sum(dim=1)
            # assert score.dim() == 1, f"{mode} score={score.shape}"
            edge_pred_dict[metapath] = score

        return edge_pred_dict

    def get_edge_index_from_neg_batch(self, pos_edges, neg_edges, mode):
        edge_index_dict = {}

        for metapath, edge_index in pos_edges.items():
            e_size, neg_samp_size = neg_edges[metapath].shape

            if mode == "head":
                nid_A = neg_edges[metapath].reshape(-1)
                nid_B = pos_edges[metapath][1].repeat_interleave(neg_samp_size)
                edge_index_dict[metapath] = torch.stack([nid_A, nid_B], dim=0)
            elif mode == "tail":
                nid_A = pos_edges[metapath][0].repeat_interleave(neg_samp_size)
                nid_B = neg_edges[metapath].reshape(-1)
                edge_index_dict[metapath] = torch.stack([nid_A, nid_B], dim=0)

        return edge_index_dict, neg_samp_size


class LATTELinkPred(LinkPredTrainer):
    def __init__(self, hparams, dataset: HeteroNetDataset, metrics=["obgl-biokg"],
                 collate_fn="neighbor_sampler") -> None:
        super(LATTELinkPred, self).__init__(hparams, dataset, metrics)
        self.head_node_type = dataset.head_node_type
        self.dataset = dataset
        self.multilabel = dataset.multilabel
        self._name = f"LATTE-{hparams.t_order}{' Link' if hparams.use_proximity else ''}"
        self.collate_fn = collate_fn

        self.embedder = LATTE(t_order=hparams.t_order, embedding_dim=hparams.embedding_dim,
                              in_channels_dict=dataset.node_attr_shape, num_nodes_dict=dataset.num_nodes_dict,
                              metapaths=dataset.get_metapaths(), attn_heads=hparams.attn_heads,
                              attn_activation=hparams.attn_activation, attn_dropout=hparams.attn_dropout,
                              use_proximity=hparams.use_proximity, neg_sampling_ratio=hparams.neg_sampling_ratio,
                              cpu_embeddings=True if "cpu_embedding" in hparams else False)

        self.classifier = DistMulti(embedding_dim=hparams.embedding_dim * hparams.t_order, metapaths=dataset.metapaths)
        self.criterion = LinkPredLoss()

        hparams.embedding_dim = hparams.embedding_dim * hparams.t_order

    def forward(self, inputs: dict, edges_true: dict, **kwargs):
        embeddings, proximity_loss, _ = self.embedder(inputs["x_dict"],
                                                      edge_index_dict=inputs["edge_index_dict"],
                                                      global_node_idx=inputs["global_node_index"],
                                                      **kwargs)

        edges_pred = self.classifier(edges_true, embeddings)

        return embeddings, proximity_loss, edges_pred

    def training_step(self, batch, batch_nb):
        X, edge_true, edge_weights = batch
        embeddings, prox_loss, edge_pred_dict = self.forward(X, edge_true)

        e_pos, e_neg, e_weights = self.reshape_e_pos_neg(edge_pred_dict, edge_weights)
        loss = self.criterion.forward(e_pos, e_neg, pos_weights=e_weights)

        self.train_metrics.update_metrics(e_pos, e_neg, weights=None)
        outputs = {'loss': loss}
        return outputs

    def validation_step(self, batch, batch_nb):
        X, edge_true, edge_weights = batch
        embeddings, prox_loss, edge_pred_dict = self.forward(X, edge_true)

        e_pos, e_neg, e_weights = self.reshape_e_pos_neg(edge_pred_dict, edge_weights)
        loss = self.criterion.forward(e_pos, e_neg, pos_weights=e_weights)

        self.valid_metrics.update_metrics(e_pos, e_neg, weights=None)
        print(F.sigmoid(e_pos[:5]).detach().cpu().numpy(), "\t",
              F.sigmoid(e_neg[:5, 0].view(-1)).detach().cpu().numpy()) if batch_nb == 1 else None

        return {"val_loss": loss}

    def test_step(self, batch, batch_nb):
        X, edge_true, edge_weights = batch
        embeddings, prox_loss, edge_pred_dict = self.forward(X, edge_true)

        e_pos, e_neg, e_weights = self.reshape_e_pos_neg(edge_pred_dict, edge_weights)
        loss = self.criterion.forward(e_pos, e_neg, pos_weights=e_weights)

        self.test_metrics.update_metrics(e_pos, e_neg, weights=None)

        return {"test_loss": loss}

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'alpha_activation', 'embedding']
        optimizer_grouped_parameters = [
            {'params': [p for name, p in param_optimizer if not any(key in name for key in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for name, p in param_optimizer if any(key in name for key in no_decay)], 'weight_decay': 0.0}
        ]

        # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-06, lr=self.hparams.lr)
        optimizer = torch.optim.Adam(optimizer_grouped_parameters,
                                     lr=self.hparams.lr,  # momentum=self.hparams.momentum,
                                     weight_decay=self.hparams.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "loss"}
