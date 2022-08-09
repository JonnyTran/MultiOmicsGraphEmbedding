import math

import dgl
from dgl import utils, DGLHeteroGraph


def copy_ndata(old_g: dgl.DGLHeteroGraph, new_g: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
    for ntype in old_g.ntypes:
        for k, v in old_g.nodes[ntype].data.items():
            new_g.nodes[ntype].data[k] = v.detach().clone()

    node_frames = utils.extract_node_subframes(new_g,
                                               nodes_or_device=[new_g.nodes(ntype) for ntype in new_g.ntypes],
                                               store_ids=True)
    utils.set_new_frames(new_g, node_frames=node_frames)
    return new_g


def dgl_to_edge_index_dict(g: DGLHeteroGraph, global_ids):
    edge_index_dict = {}
    for metapath in g.canonical_etypes:
        head_type, etype, tail_type = metapath
        u, v = g.edges(etype=metapath[1], )
        if len(u) == 0: continue

        if global_ids:
            u = g.nodes[head_type].data["_ID"][u]
            v = g.nodes[tail_type].data["_ID"][v]

        edge_index_dict[metapath] = (u, v)

    return edge_index_dict


def round_to_multiple(number, multiple):
    num = int(multiple * math.floor(number / multiple))
    return num
