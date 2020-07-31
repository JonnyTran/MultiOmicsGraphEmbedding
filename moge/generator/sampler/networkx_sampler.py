import copy
import multiprocessing
from collections import OrderedDict
from itertools import islice

import networkx as nx
import numpy as np
import torch

from moge.generator.datasets import HeteroNetDataset


class NetworkXSampler(HeteroNetDataset):
    def __init__(self, dataset, node_types, metapaths=None, head_node_type=None, directed=True, train_ratio=0.7,
                 add_reverse_metapaths=True, multiworker=True, process_graphs=True):
        self.multiworker = multiworker
        super().__init__(dataset, node_types, metapaths, head_node_type, directed, train_ratio, add_reverse_metapaths,
                         process_graphs)
        self.char_to_node_type = {node_type[0]: node_type for node_type in self.node_types}

    def process_graph_sampler(self):
        try:
            cpus = multiprocessing.cpu_count()
        except NotImplementedError:
            cpus = 1  # arbitrary default

        if self.multiworker:
            self.graphs = {}
            pool = multiprocessing.Pool(processes=cpus)
            output = pool.map(self.create_nx_graph, self.metapaths)
            for (metapath, graph) in output:
                self.graphs[metapath] = graph
            pool.close()
        else:
            self.graphs = {metapath: graph for metapath, graph in
                           [self.create_nx_graph(metapath) for metapath in self.metapaths]}

        self.join_graph = nx.compose_all([G.to_undirected() for metapath, G in self.graphs.items()])

    def create_nx_graph(self, metapath):
        """
        A hashable function that forks a
        :param metapath:
        :return: (metapath:str, graph:networkx.Graph)
        """
        edgelist = self.edge_index_dict[metapath].t().numpy().astype(str)
        edgelist = np.core.defchararray.add([metapath[0][0], metapath[-1][0]], edgelist)
        graph = nx.from_edgelist(edgelist, create_using=nx.DiGraph if self.directed else nx.Graph)
        return (metapath, graph)

    def neighbors_traversal(self, batch_size: int, seed_nodes: {str: list}, max_iter=3):
        num_node_all = 0
        i = 0
        while num_node_all < batch_size and i < max_iter:
            for metapath, G in self.graphs.items():
                head_type, tail_type = metapath[0], metapath[-1]

                if head_type in seed_nodes and len(seed_nodes[head_type][-1]) > 0:
                    neighbor_type = tail_type
                    source_type = head_type
                elif tail_type in seed_nodes and len(seed_nodes[tail_type][-1]) > 0:
                    neighbor_type = head_type
                    source_type = tail_type
                else:
                    continue
                if neighbor_type == self.head_node_type: continue

                source_nodes = [node for node in seed_nodes[source_type][-1] if node in G]
                neighbors = [neighbor for source in source_nodes for neighbor in nx.neighbors(G, source)]

                # Ensure that no node_type becomes the majority of the batch_size
                if len(neighbors) > (batch_size / ((i + 1) * len(self.node_types))):
                    np.random.shuffle(neighbors)
                    neighbors = neighbors[: int(batch_size / ((i + 1) * len(self.node_types)))]

                # Add neighbors nodes as a set to the sets in seed_nodes[neighbor_type]
                seed_nodes.setdefault(neighbor_type, []).append(neighbors)

            # Check whether to gather more nodes to fill batch_size
            num_node_all = sum([len(nodes) for node_type, node_sets in seed_nodes.items() for nodes in node_sets])
            i += 1

        # Join all sampled node list in each node type
        sampled_nodes = {}
        for node_type, lists in seed_nodes.items():
            sampled_nodes[node_type] = [node for nodelist in lists for node in nodelist]

        # Remove duplicate
        for node_type in sampled_nodes.keys():
            sampled_nodes[node_type] = list(OrderedDict.fromkeys(sampled_nodes[node_type]))

        # Remove excess node if exceeds batch_size
        if num_node_all > batch_size:
            largest_node_type = max({k: v for k, v in sampled_nodes.items()}, key=len)
            np.random.shuffle(sampled_nodes[largest_node_type])
            num_node_remove = num_node_all - batch_size
            sampled_nodes[largest_node_type] = sampled_nodes[largest_node_type][:-num_node_remove]

        return sampled_nodes

    def bfs_traversal(self, batch_size: int, seed_nodes: list, traversal_depth=2):
        sampled_nodes = copy.copy(seed_nodes)

        while len(sampled_nodes) < batch_size:
            for start_node in reversed(sampled_nodes):
                if start_node is None or start_node not in self.join_graph:
                    continue
                successor_nodes = [node for source, successors in
                                   islice(nx.traversal.bfs_successors(self.join_graph,
                                                                      source=start_node), traversal_depth) for node in
                                   successors]
                if len(successor_nodes) > (batch_size / (2 * len(self.node_types))):
                    successor_nodes = successor_nodes[:int(batch_size / (2 * len(self.node_types)))]
                sampled_nodes.extend(successor_nodes)

            sampled_nodes = list(OrderedDict.fromkeys(sampled_nodes))

        if len(sampled_nodes) > batch_size:
            np.random.shuffle(sampled_nodes)
            sampled_nodes = sampled_nodes[:batch_size]

        # Sort sampled nodes to node_name_dict
        node_index_dict = {}
        for node in sampled_nodes:
            node_index_dict.setdefault(self.char_to_node_type[node[0]], []).append(node)
        node_index_dict[self.head_node_type] = seed_nodes
        return node_index_dict

    def get_adj_edgelist(self, graph, nodes_A, nodes_B=None):
        if nodes_B == None:
            adj = nx.adj_matrix(graph, nodelist=nodes_A.numpy() if isinstance(nodes_A, torch.Tensor) else nodes_A)
        else:
            adj = nx.algorithms.bipartite.biadjacency_matrix(graph, row_order=nodes_A, column_order=nodes_B)

        adj = adj.tocoo()
        edge_index = torch.tensor(np.vstack([adj.row, adj.col]), dtype=torch.long)
        return edge_index

    def convert_index2name(self, node_idx: torch.Tensor, node_type: str):
        return np.core.defchararray.add(node_type[0], node_idx.numpy().astype(str)).tolist()

    def strip_node_type_str(self, node_names: list):
        """
        Strip letter from node names
        :param node_names:
        :return:
        """
        return torch.tensor([int(name[1:]) for name in node_names], dtype=torch.long)

    def convert_sampled_nodes_to_node_dict(self, node_names: list):
        """
        Strip letter from node names
        :param node_names:
        :return:
        """
        node_index_dict = {}
        for node in node_names:
            node_index_dict.setdefault(self.char_to_node_type[node[0]], []).extend([int(node[1:])])

        node_index_dict = {k: torch.tensor(v, dtype=torch.long) for k, v in node_index_dict.items()}
        return node_index_dict

    def get_collate_fn(self, collate_fn: str, batch_size=None):
        if batch_size is not None:
            self.batch_size = batch_size * len(self.node_types)

        if "LATTENode_batch" in collate_fn:
            return self.collate_LATTENode_batch
        else:
            raise Exception(f"Correct collate function {collate_fn} not found.")

    def collate_LATTENode_batch(self, iloc):
        if not isinstance(iloc, torch.Tensor):
            iloc = torch.tensor(iloc)

        sampled_nodes = self.bfs_traversal(batch_size=self.batch_size,
                                           seed_nodes=self.convert_index2name(iloc, self.head_node_type))
        # node_index = self.y_index_dict[self.head_node_type][iloc]
        # print("sampled_nodes", {k: len(v) for k, v in sampled_nodes.items()})
        assert len(iloc) == len(sampled_nodes[self.head_node_type])
        X = {"edge_index_dict": {}, "global_node_index": {}, "x_dict": {}}

        for metapath in self.metapaths:
            head_type, tail_type = metapath[0], metapath[-1]
            if head_type not in sampled_nodes or len(sampled_nodes[head_type]) == 0: continue
            if tail_type not in sampled_nodes or len(sampled_nodes[tail_type]) == 0: continue
            try:
                X["edge_index_dict"][metapath] = self.get_adj_edgelist(self.graphs[metapath],
                                                                       nodes_A=sampled_nodes[head_type],
                                                                       nodes_B=sampled_nodes[tail_type])
            except Exception as e:
                print("sampled_nodes[head_type]", sampled_nodes[head_type])
                print("sampled_nodes[tail_type]", sampled_nodes[tail_type])
                raise e

        if self.use_reverse:
            self.add_reverse_edge_index(X["edge_index_dict"])

        for node_type in sampled_nodes:
            X["global_node_index"][node_type] = self.strip_node_type_str(sampled_nodes[node_type])

        if hasattr(self, "x_dict"):
            X["x_dict"] = {node_type: self.x_dict[node_type][X["global_node_index"][node_type]] for node_type in
                           self.x_dict}

        if len(self.y_dict) > 1:
            y = {node_type: y_true[X["global_node_index"][node_type]] for node_type, y_true in self.y_dict.items()}
        else:
            y = self.y_dict[self.head_node_type][iloc].squeeze(-1)
        return X, y, None
