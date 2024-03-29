# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import dgl
import numpy as np
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, default=str)
parser.add_argument("--method", required=True, default=str)
parser.add_argument("--emb-file", type=str, default=None)
parser.add_argument("--root_path", required=False, default="/home/jonny/Bioinformatics_ExternalData/OGB/")
args = parser.parse_args()

if args.emb_file is None:
    # default path
    args.emb_file = (
        f"ckpts/{args.method}_{args.dataset}_0/{args.dataset}_{args.method}_entity.npy"
    )
    print(f"Using default path of node embedding file: {args.emb_file}")

emb = np.load(args.emb_file)

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

node_type = g.ntypes if hasattr(g, "ntypes") else ["_N"]
node_offset = [0]
for ntype in node_type:
    num_nodes = g.number_of_nodes(ntype) if isinstance(g, dgl.DGLHeteroGraph) else g.number_of_nodes()
    node_offset.append(num_nodes + node_offset[-1])

# reorder embedding to original node order
real_emb = np.zeros((node_offset[-1], emb.shape[1]))
real_id = []
with open(f"entities.tsv") as f:
    for line in f:
        tokens = line.strip().split("\t")
        real_id.append(int(tokens[1]))
real_id = np.array(real_id)
real_emb[real_id] = emb

for i, ntype in enumerate(node_type):
    node_emb = torch.from_numpy(real_emb[node_offset[i]: node_offset[i + 1]])
    torch.save(node_emb, f"{ntype}.pt")
