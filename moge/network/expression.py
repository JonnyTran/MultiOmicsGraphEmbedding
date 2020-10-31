from moge.network.multi_digraph import MultiDigraphNetwork
from moge.network.semantic_similarity import *


@DeprecationWarning
class ExpressionNetwork(MultiDigraphNetwork):

    def add_sampled_undirected_negative_edges_from_correlation(self, modalities=[], correlation_threshold=0.2,
                                                               histological_subtypes=[],
                                                               pathologic_stages=[]):
        """
        Sample edges with experssion values absolute-value correlations near zero, indicating no relationships

        :param modalities:
        :param correlation_threshold:
        :return:
        """
        nodes_A = self.nodes[modalities[0]]
        nodes_B = self.nodes[modalities[1]]
        node_list = [node for node in self.node_list if node in nodes_A or node in nodes_B]

        # Filter similarity adj by correlation
        correlation_dist = compute_expression_correlation(self.multiomics, modalities=modalities,
                                                          node_list=node_list,
                                                          histological_subtypes=histological_subtypes,
                                                          pathologic_stages=pathologic_stages,
                                                          squareform=True)
        correlation_dist = 1 - correlation_dist
        correlation_dist = np.abs(correlation_dist)

        similarity_filtered = np.triu(correlation_dist <= correlation_threshold, k=1)  # A True/False matrix
        sim_edgelist_ebunch = [(node_list[x], node_list[y], {"weight": correlation_dist.iloc[x, y]}) for x, y in
                               zip(*np.nonzero(similarity_filtered))]
        print(sim_edgelist_ebunch[0:10])
        self.G.add_weighted_edges_from(sim_edgelist_ebunch, type="u_n")
        print(len(sim_edgelist_ebunch), "undirected positive edges (type='u') added.")
