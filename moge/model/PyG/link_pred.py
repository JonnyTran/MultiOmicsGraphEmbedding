import logging
from typing import List, Tuple, Dict, Any

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau

from moge.model.PyG.latte_flat import LATTE
from moge.model.losses import LinkPredLoss
from ..encoder import HeteroSequenceEncoder, HeteroNodeEncoder
from ..trainer import LinkPredTrainer
from ...dataset import HeteroLinkPredDataset


class DistMulti(torch.nn.Module):
    def __init__(self, embedding_dim: int, metapaths: List[Tuple[str, str, str]]):
        """

        Args:
            embedding_dim ():
            metapaths (): List of metapaths to predict
        """
        super(DistMulti, self).__init__()
        self.metapaths = metapaths
        self.embedding_dim = embedding_dim

        # self.linears = nn.ModuleDict(
        #     {metapath: nn.Parameter(torch.Tensor((embedding_dim, embedding_dim))) \
        #      for metapath in metapaths}
        # )

        self.relation_embedding = nn.Parameter(torch.zeros(len(metapaths), embedding_dim), requires_grad=True)
        nn.init.uniform_(tensor=self.relation_embedding, a=-1, b=1)

    def forward(self, inputs: Dict[str, Dict[Tuple[str, str, str], Tensor]],
                embeddings: Dict[str, Tensor]) -> Dict[str, Dict[Tuple[str, str, str], Tensor]]:
        output = {}
        # pprint(tensor_sizes({"inputs":inputs, "embeddings":embeddings}), width=200)
        # print("edge_pos", {m: inputs["edge_pos"][m].max(1).values for m in inputs["edge_pos"]})
        # print("head-batch", {m: torch.max(inputs["head-batch"][m]) for m in inputs["head-batch"]})
        # print("tail-batch", {m: torch.max(inputs["tail-batch"][m]) for m in inputs["tail-batch"]})

        # Single edges
        output["edge_pos"] = self.predict(inputs["edge_pos"], embeddings, mode="single")

        # Sampled head or tail negative sampling
        if "head-batch" in inputs or "tail-batch" in inputs:
            # Head batch
            edge_head_batch = self.get_edge_index_from_neg_batch(inputs["edge_pos"],
                                                                 neg_edges=inputs["head-batch"],
                                                                 mode="head")
            output["head-batch"] = self.predict(edge_head_batch, embeddings, mode="head")

            # Tail batch
            edge_tail_batch = self.get_edge_index_from_neg_batch(inputs["edge_pos"],
                                                                 neg_edges=inputs["tail-batch"],
                                                                 mode="tail")
            output["tail-batch"] = self.predict(edge_tail_batch, embeddings, mode="tail")

        # True negative edges
        elif "edge_neg" in inputs:
            output["edge_neg"] = self.predict(edge_index_dict=inputs["edge_neg"], embeddings=embeddings, mode="single")
        else:
            raise Exception(f"No negative edges in inputs {inputs.keys()}")

        return output

    def predict(self, edge_index_dict: Dict[Tuple[str, str, str], Tensor],
                embeddings: Dict[str, Tensor],
                mode: str) -> Dict[Tuple[str, str, str], Tensor]:
        edge_pred_dict = {}

        for metapath, edge_index in edge_index_dict.items():
            metapath_idx = self.metapaths.index(metapath)
            head_type, edge_type, tail_type = metapath

            kernel = self.relation_embedding[metapath_idx]  # (emb_dim)

            if "tail" == mode:
                side_A = (embeddings[head_type] * kernel)[edge_index[0]].unsqueeze(1)  # (n_nodes, 1, emb_dim)
                emb_B = embeddings[tail_type][edge_index[1]].unsqueeze(2)  # (n_nodes, emb_dim, 1)
                # score = side_A * emb_B
                score = torch.bmm(side_A, emb_B)
                score = score.sum(-1)
            else:
                emb_A = embeddings[head_type][edge_index[0]].unsqueeze(1)  # (n_nodes, 1, emb_dim)
                side_B = (kernel * embeddings[tail_type][edge_index[1]]).unsqueeze(2)  # (n_nodes, emb_dim, 1)

                # score = emb_A * side_B
                score = torch.bmm(emb_A, side_B)
                score = score.sum(-1)

            score = score.sum(dim=1)
            # assert score.dim() == 1, f"{mode} score={score.shape}"
            edge_pred_dict[metapath] = score

        return edge_pred_dict

    def get_edge_index_from_neg_batch(self, pos_edges: Dict[Tuple[str, str, str], Tensor],
                                      neg_edges: Dict[Tuple[str, str, str], Tensor],
                                      mode: str) -> Dict[Tuple[str, str, str], Tensor]:
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

        return edge_index_dict


class LATTELinkPred(LinkPredTrainer):
    def __init__(self, hparams, dataset: HeteroLinkPredDataset, metrics=["obgl-biokg"],
                 collate_fn="neighbor_sampler") -> None:
        super().__init__(hparams, dataset, metrics)
        self.head_node_type = dataset.head_node_type
        self.dataset = dataset
        self.multilabel = dataset.multilabel
        self._name = f"LATTE-{hparams.t_order} Link"
        self.collate_fn = collate_fn

        # Node attr input
        if hasattr(dataset, 'seq_tokenizer'):
            self.encoder = HeteroSequenceEncoder(hparams, dataset)
        else:
            self.encoder = HeteroNodeEncoder(hparams, dataset)

        self.embedder = LATTE(n_layers=hparams.n_layers,
                              t_order=min(hparams.t_order, hparams.n_layers),
                              embedding_dim=hparams.embedding_dim,
                              num_nodes_dict=dataset.num_nodes_dict,
                              metapaths=dataset.get_metapaths(khop=True if "khop" in collate_fn else None),
                              layer_pooling=hparams.layer_pooling,
                              activation=hparams.activation,
                              attn_heads=hparams.attn_heads,
                              attn_activation=hparams.attn_activation,
                              attn_dropout=hparams.attn_dropout,
                              use_proximity=hparams.use_proximity if hasattr(hparams, "use_proximity") else False,
                              neg_sampling_ratio=hparams.neg_sampling_ratio \
                                  if hasattr(hparams, "neg_sampling_ratio") else None,
                              edge_sampling=hparams.edge_sampling if hasattr(hparams, "edge_sampling") else False,
                              hparams=hparams)
        if hparams.layer_pooling == "concat":
            hparams.embedding_dim = hparams.embedding_dim * hparams.t_order
            logging.info("embedding_dim {}".format(hparams.embedding_dim))

        self.classifier = DistMulti(embedding_dim=hparams.embedding_dim, metapaths=dataset.pred_metapaths)
        self.criterion = LinkPredLoss()

    def forward(self, inputs: Dict[str, Any], edges_true: Dict[str, Dict[Tuple[str, str, str], Tensor]], **kwargs) \
            -> Tuple[Dict[str, Tensor], Any, Dict[str, Dict[Tuple[str, str, str], Tensor]]]:
        if not self.training:
            self._node_ids = inputs["global_node_index"]

        if "sequences" in inputs and isinstance(self.encoder, HeteroSequenceEncoder):
            h_out = self.encoder.forward(inputs['sequences'])
        elif "x_dict" in inputs:
            h_out = self.encoder.forward(inputs["x_dict"], global_node_idx=inputs["global_node_index"])

        embeddings, aux_loss, _ = self.embedder.forward(h_out,
                                                        edge_index_dict=inputs["edge_index_dict"],
                                                        global_node_idx=inputs["global_node_index"],
                                                        sizes=inputs["sizes"],
                                                        **kwargs)

        edges_pred = self.classifier.forward(edges_true, embeddings)

        return embeddings, aux_loss, edges_pred

    def training_step(self, batch, batch_nb):
        X, edge_true, edge_weights = batch
        embeddings, prox_loss, edge_pred_dict = self.forward(X, edge_true)

        e_pos, e_neg, e_weights = self.reshape_e_pos_neg(edge_pred_dict, edge_weights)
        loss = self.criterion.forward(e_pos, e_neg, pos_weights=e_weights)

        self.train_metrics.update_metrics(e_pos, e_neg, weights=None)

        logs = {'loss': loss, **self.train_metrics.compute_metrics()}
        self.log_dict(logs, prog_bar=True, logger=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_nb):
        X, edge_true, edge_weights = batch
        embeddings, prox_loss, edge_pred_dict = self.forward(X, edge_true)

        e_pos, e_neg, e_weights = self.reshape_e_pos_neg(edge_pred_dict, edge_weights)
        loss = self.criterion.forward(e_pos, e_neg, pos_weights=e_weights)

        self.valid_metrics.update_metrics(e_pos, e_neg, weights=None)
        print(F.sigmoid(e_pos[:5]).detach().cpu().numpy(), "\t",
              F.sigmoid(e_neg[:5, 0].view(-1)).detach().cpu().numpy()) if batch_nb == 1 else None
        self.log("val_loss", loss)

        return loss

    def test_step(self, batch, batch_nb):
        X, edge_true, edge_weights = batch
        embeddings, prox_loss, edge_pred_dict = self.forward(X, edge_true)

        e_pos, e_neg, e_weights = self.reshape_e_pos_neg(edge_pred_dict, edge_weights)
        loss = self.criterion.forward(e_pos, e_neg, pos_weights=e_weights)
        self.test_metrics.update_metrics(e_pos, e_neg, weights=None)
        # print(tensor_sizes({"e_pos": e_pos, "e_neg": e_neg}))
        # return

        self.log("test_loss", loss)
        return loss

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
