# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import dgl
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, default=str)
parser.add_argument("--root_path", required=False, default="/home/jonny/Bioinformatics_ExternalData/OGB/")
args = parser.parse_args()

if "ogbn" in args.dataset:
    from ogb.nodeproppred import DglNodePropPredDataset

    home_dir = os.getenv("HOME")
    dataset = DglNodePropPredDataset(
        name=args.dataset, root=args.root_path if "root_path" in args else os.path.join(home_dir, ".ogb", "dataset")
    )
    g, _ = dataset[0]
elif args.dataset == "acm":
    import sys

    sys.path.append("..")
    from data import load_acm_raw

    dataset = load_acm_raw()
    g = dataset[0]
elif args.dataset.startswith("oag"):
    import pickle

    if args.dataset == "oag_L1":
        graph_file = "../oag_dataset/graph_L1.pk"
    if args.dataset == "oag_venue":
        graph_file = "../oag_dataset/graph_venue.pk"
    with open(graph_file, "rb") as f:
        dataset = pickle.load(f)
    g = dgl.heterograph(dataset["edges"])
else:
    print(f"Dataset {args.dataset} not supported")
    exit(-1)

node_types = g.ntypes if hasattr(g, "ntypes") else ["_N"]
node_offset = [0]
for ntype in node_types:
    num_nodes = g.number_of_nodes(ntype) if isinstance(g, dgl.DGLHeteroGraph) else g.number_of_nodes()
    node_offset.append(num_nodes + node_offset[-1])

node_offset = node_offset[:-1]

with open(f"train_triplets_{args.dataset}", "w") as f:
    edge_types = g.etypes if hasattr(g, "etypes") else ["_E"]
    for etype in edge_types:
        stype, _, dtype = g.to_canonical_etype(etype) if isinstance(g, dgl.DGLHeteroGraph) else ("_N", "_E", "_N")
        src, dst = g.all_edges(etype=etype) if isinstance(g, dgl.DGLHeteroGraph) else g.all_edges()
        src = src.numpy() + node_offset[node_types.index(stype)]
        dst = dst.numpy() + node_offset[node_types.index(dtype)]

        for u, v in zip(src, dst):
            f.write("{}\t{}\t{}\n".format(u, etype, v))
