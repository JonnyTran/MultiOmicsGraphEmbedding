from typing import List, Tuple, Dict, Union

from dgl import BaseTransform, convert, DGLHeteroGraph
from dgl.heterograph import DGLBlock
from dgl.transforms.module import update_graph_structure
from scipy.sparse import csr_matrix


class AddMetaPaths(BaseTransform):
    def __init__(self, metapaths: Dict[str, List[Tuple[str, str, str]]], keep_orig_edges=True):
        self.metapaths = metapaths
        self.keep_orig_edges = keep_orig_edges

    def __call__(self, g: Union[DGLHeteroGraph]):
        data_dict = dict()

        for meta_etype, metapath_chain in self.metapaths.items():
            meta_g = metapath_reachable_graph(g, metapath_chain)
            u_type = metapath_chain[0][0]
            v_type = metapath_chain[-1][-1]
            data_dict[(u_type, meta_etype, v_type)] = meta_g.edges()

        if self.keep_orig_edges:
            for c_etype in g.canonical_etypes:
                data_dict[c_etype] = g.edges(etype=c_etype)
            new_g = update_graph_structure(g, data_dict, copy_edata=True)
        else:
            new_g = update_graph_structure(g, data_dict, copy_edata=False)

        return new_g


def metapath_reachable_graph(g: DGLHeteroGraph, metapath_chain: List[Tuple[str, str, str]]):
    adj = 1
    for etype in metapath_chain:
        adj = adj * g.adj(etype=etype, scipy_fmt='csr', transpose=False)

    adj = (adj != 0).tocsr()
    srctype = g.to_canonical_etype(metapath_chain[0])[0]
    dsttype = g.to_canonical_etype(metapath_chain[-1])[2]
    new_g = convert.heterograph({(srctype, '_E', dsttype): adj.nonzero()},
                                {srctype: adj.shape[0], dsttype: adj.shape[1]},
                                idtype=g.idtype, device=g.device)

    # copy srcnode features
    new_g.nodes[srctype].data.update(g.nodes[srctype].data)
    # copy dstnode features
    if srctype != dsttype:
        new_g.nodes[dsttype].data.update(g.nodes[dsttype].data)

    return new_g


def get_larger_block(src_block: DGLBlock, dst_block: DGLBlock, ntype: str) -> DGLBlock:
    src_dst_sizes = dict(zip([src_block, dst_block], [src_block.num_nodes(ntype), dst_block.num_nodes(ntype)]))
    block = max(src_dst_sizes, key=src_dst_sizes.get)
    return block


def metapath_reachable_blocks(src_block: DGLBlock, dst_block: DGLBlock,
                              metapath_chain: List[Tuple[str, str, str]]):
    adj: csr_matrix = 1
    for block, etype in zip([src_block, dst_block], metapath_chain):
        next_adj = block.adj(etype=etype, scipy_fmt='csr', transpose=False)

        if isinstance(adj, csr_matrix) and adj.shape[1] > next_adj.shape[0]:
            adj = adj[:, :next_adj.shape[0]]
        adj = adj * next_adj

    adj = (adj != 0).tocsr()
    srctype = src_block.to_canonical_etype(metapath_chain[0])[0]
    dsttype = dst_block.to_canonical_etype(metapath_chain[-1])[2]

    num_nodes_dict = {srctype: src_block.num_nodes(srctype), dsttype: src_block.num_nodes(dsttype)}
    src_feats = src_block.nodes[srctype].data

    new_g: DGLHeteroGraph = convert.heterograph({(srctype, '_E', dsttype): adj.nonzero()},
                                                num_nodes_dict=num_nodes_dict,
                                                idtype=src_block.idtype, device=src_block.device)
    # copy srcnode features
    new_g.nodes[srctype].data.update(src_feats)
    # copy dstnode features
    if srctype != dsttype:
        dst_feats = src_block.nodes[dsttype].data
        new_g.nodes[dsttype].data.update(dst_feats)

    return new_g


class ChainMetaPaths(BaseTransform):
    def __init__(self, metapaths: Dict[str, List[Tuple[str, str, str]]], keep_orig_edges=True):
        self.metapaths = metapaths
        self.keep_orig_edges = keep_orig_edges

    def __call__(self, src_block: DGLBlock, dst_block: DGLBlock):
        data_dict = dict()

        for meta_etype, metapath_chain in self.metapaths.items():
            meta_g = metapath_reachable_blocks(src_block, dst_block, metapath_chain)
            u_type = metapath_chain[0][0]
            v_type = metapath_chain[-1][-1]
            data_dict[(u_type, meta_etype, v_type)] = meta_g.edges()

        if self.keep_orig_edges:
            for c_etype in src_block.canonical_etypes:
                data_dict[c_etype] = src_block.edges(etype=c_etype)
            new_g = update_graph_structure(src_block, data_dict, copy_edata=False)
        else:
            new_g = update_graph_structure(src_block, data_dict, copy_edata=False)

        return new_g



