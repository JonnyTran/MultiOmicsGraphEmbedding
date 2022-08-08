import dgl
from dgl import utils


def copy_ndata(old_g: dgl.DGLHeteroGraph, new_g: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
    for ntype in old_g.ntypes:
        for k, v in old_g.nodes[ntype].data.items():
            new_g.nodes[ntype].data[k] = v.detach().clone()

    node_frames = utils.extract_node_subframes(new_g,
                                               nodes_or_device=[new_g.nodes(ntype) for ntype in new_g.ntypes],
                                               store_ids=True)
    utils.set_new_frames(new_g, node_frames=node_frames)
    return new_g
