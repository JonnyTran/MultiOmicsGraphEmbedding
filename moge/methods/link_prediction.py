import torch
import pytorch_lightning as pl

from .node_classification import NodeClfMetrics, LATTENodeClassifier
from ..generator.datasets import HeteroNetDataset
from ..generator.sampler.link_sampler import LinkSampler
from ..module.latte import LATTE


class LinkPredictionMetrics(NodeClfMetrics):
    def __init__(self):
        super(LinkPredictionMetrics, self).__init__()


class LATTELinkPredictor(LATTENodeClassifier):
    def __init__(self, hparams, dataset: HeteroNetDataset, metrics=["obgl-biokg"],
                 collate_fn="neighbor_sampler") -> None:
        super().__init__(hparams, dataset, metrics, collate_fn)

    def forward(self, x_dict, global_node_index, edge_index_dict):
        embeddings, proximity_loss = self.latte.forward(x_dict, global_node_index, edge_index_dict)
        y_hat = self.classifier.forward(embeddings[self.head_node_type])
        return y_hat, proximity_loss
