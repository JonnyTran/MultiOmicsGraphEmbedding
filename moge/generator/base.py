import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch_sparse


class Network:
    def get_networkx(self):
        if not hasattr(self, "G"):
            G = nx.Graph()
            for metapath in self.edge_index_dict:
                edgelist = self.edge_index_dict[metapath].t().numpy().astype(str)
                edgelist = np.core.defchararray.add([metapath[0][0], metapath[-1][0]], edgelist)
                edge_type = "".join([n for i, n in enumerate(metapath) if i % 2 == 1])
                G.add_edges_from(edgelist, edge_type=edge_type)

            self.G = G

        return self.G

    def get_projection_pos(self, embeddings_all, UMAP: classmethod, n_components=2):
        pos = UMAP(n_components=n_components).fit_transform(embeddings_all)
        pos = {embeddings_all.index[i]: pair for i, pair in enumerate(pos)}
        return pos

    def get_node_degrees(self, directed=True):
        index = pd.concat([pd.DataFrame(range(v), [k, ] * v) for k, v in self.num_nodes_dict.items()],
                          axis=0).reset_index()
        multi_index = pd.MultiIndex.from_frame(index, names=["node_type", "node"])

        metapaths = list(self.edge_index_dict.keys())
        metapath_names = [".".join(metapath) if isinstance(metapath, tuple) else metapath for metapath in
                          metapaths]
        self.node_degrees = pd.DataFrame(data=0, index=multi_index,
                                         columns=metapath_names)

        for metapath, name in zip(metapaths, metapath_names):
            edge_index = self.edge_index_dict[metapath]

            head, tail = metapath[0], metapath[-1]
            D = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1],
                                          sparse_sizes=(self.num_nodes_dict[head],
                                                        self.num_nodes_dict[tail]))

            self.node_degrees.loc[(head, name)] = (
                    self.node_degrees.loc[(head, name)] + D.storage.rowcount().numpy()).values
            if not directed:
                self.node_degrees.loc[(tail, name)] = (
                        self.node_degrees.loc[(tail, name)] + D.storage.colcount().numpy()).values

        return self.node_degrees

    def get_embedding_dfs(self, embeddings_dict, global_node_index):
        embeddings = []
        for node_type in self.node_types:
            nodes = global_node_index[node_type].numpy().astype(str)
            nodes = np.core.defchararray.add(node_type[0], nodes)
            if isinstance(embeddings_dict[node_type], torch.Tensor):
                df = pd.DataFrame(embeddings_dict[node_type].detach().cpu().numpy(), index=nodes)
            else:
                df = pd.DataFrame(embeddings_dict[node_type], index=nodes)
            embeddings.append(df)

        return embeddings

    def get_embeddings_types_labels(self, embeddings, global_node_index):
        embeddings_all = pd.concat(embeddings, axis=0)

        types_all = embeddings_all.index.to_series().str.slice(0, 1)
        if hasattr(self, "y_dict") and len(self.y_dict) > 0:
            labels = pd.Series(
                self.y_dict[self.head_node_type][global_node_index[self.head_node_type]].squeeze(-1).numpy(),
                index=embeddings[0].index,
                dtype=str)
        else:
            labels = None

        return embeddings_all, types_all, labels
