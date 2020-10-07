import torch
from ogb.linkproppred import PygLinkPropPredDataset

from moge.module.PyG.latte import tag_negative
from .. import HeteroNetDataset


class EdgeSampler(HeteroNetDataset):
    def __init__(self, dataset, node_types=None, metapaths=None, head_node_type=None, directed=True,
                 resample_train=None,
                 add_reverse_metapaths=True):
        super(EdgeSampler, self).__init__(dataset, node_types, metapaths, head_node_type, directed, resample_train,
                                          add_reverse_metapaths)

    def process_PygLinkDataset_homo(self, dataset: PygLinkPropPredDataset):
        data = dataset[0]
        self._name = dataset.name

        self.head_node_type = "entity"
        self.metapaths = [(self.head_node_type, "default", self.head_node_type)]

        self.edge_index_dict = {self.metapaths[0]: data.edge_index}
        self.num_nodes_dict = self.get_num_nodes_dict(self.edge_index_dict)
        self.node_types = list(self.num_nodes_dict.keys())

        if hasattr(data, "x") and data.x is not None:
            self.x_dict = {self.head_node_type: data.x}
        elif hasattr(data, "x_dict") and data.x_dict is not None:
            self.x_dict = data.x_dict
        else:
            self.x_dict = {}
        self.node_attr_shape = {node_type: x.size(1) for node_type, x in self.x_dict.items()}

        split_idx = dataset.get_edge_split()
        train_triples, valid_triples, test_triples = split_idx["train"], split_idx["valid"], split_idx["test"]

        self.edge_index = []
        self.edge_index.extend([valid_triples["edge"], valid_triples["edge_neg"]])
        self.edge_index.extend([test_triples["edge"], test_triples["edge_neg"]])
        self.edge_index.extend([train_triples["edge"]])
        self.edge_index = torch.cat(self.edge_index, dim=0)

        self.edge_reltype = []
        self.edge_reltype.extend([torch.ones(valid_triples["edge"].size(0)),  # Ones correspond to pos edges
                                  torch.zeros(valid_triples["edge_neg"].size(0)),  # Zeroes correspond to neg edges
                                  torch.ones(test_triples["edge"].size(0)),
                                  torch.zeros(test_triples["edge_neg"].size(0)),
                                  torch.ones(train_triples["edge"].size(0))])
        self.edge_reltype = torch.cat(self.edge_reltype, dim=0).to(torch.int)

        # Build train/test/valid idx
        self.start_idx = {"valid": 0,
                          "test": len(valid_triples["edge"]) + len(valid_triples["edge_neg"])}
        self.start_idx["train"] = self.start_idx["test"] + len(test_triples["edge"]) + len(test_triples["edge_neg"])

        self.validation_idx = torch.arange(self.start_idx["valid"], self.start_idx["test"])
        self.testing_idx = torch.arange(self.start_idx["test"], self.start_idx["train"])
        self.training_idx = torch.arange(self.start_idx["train"],
                                         self.start_idx["train"] + train_triples["edge"].size(0))

        assert self.validation_idx.max() < self.testing_idx.min()
        assert self.testing_idx.max() < self.training_idx.min()

    def get_collate_fn(self, collate_fn: str, batch_size=None, mode=None):
        return self.sample

    def sample(self, iloc):
        if not isinstance(iloc, torch.Tensor):
            iloc = torch.tensor(iloc)

        X = {"edge_index_dict": {}, "global_node_index": {}, "x_dict": {}}

        edge_index = self.edge_index[iloc]
        edge_reltype = self.edge_reltype[iloc]
        reltype_ids = self.edge_reltype[iloc].unique()

        # Gather all nodes sampled
        X["global_node_index"][self.head_node_type] = edge_index.view(-1).unique()

        local2batch = {
            node_type: dict(zip(X["global_node_index"][node_type].numpy(),
                                range(len(X["global_node_index"][node_type])))
                            ) for node_type in X["global_node_index"]}

        # Get edge_index with batch id
        for relation_id in reltype_ids:
            if relation_id == 1:
                mask = edge_reltype == relation_id
                X["edge_index_dict"][self.metapaths[0]] = edge_index[mask, :].apply_(
                    local2batch[self.head_node_type].get).t()
            elif relation_id == 0:
                mask = edge_reltype == relation_id
                X["edge_index_dict"][tag_negative(self.metapaths[0])] = edge_index[mask, :].apply_(
                    local2batch[self.head_node_type].get).t()

        if self.use_reverse:
            self.add_reverse_edge_index(X["edge_index_dict"])

        # Make x_dict
        if hasattr(self, "x_dict") and len(self.x_dict) > 0:
            X["x_dict"] = {node_type: self.x_dict[node_type][X["global_node_index"][node_type]] \
                           for node_type in self.x_dict}

        return X, None, None
