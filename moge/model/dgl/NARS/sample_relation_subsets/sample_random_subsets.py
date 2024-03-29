# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import random
import time
import dgl
import networkx as nx


def sample_relation_subsets(g: nx.MultiDiGraph, args):
    # each relation has prob 0.5 to be kept
    prob = 0.5
    edges = []
    for u, v, e in g.edges:
        edges.append((u, v, e))
    n_edges = len(edges)

    subsets = set()
    max_loop = 0
    while len(subsets) < args.num_subsets and max_loop < 50:
        selected = []
        for e in edges:
            if random.random() < prob:
                selected.append(e)

        # retry if no edge is selected
        if len(selected) == 0:
            continue

        sorted(selected)
        subsets.add(tuple(selected))
        max_loop += 1

    return subsets


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Random relation subset generator")
    parser.add_argument("--num-subsets", type=int, required=True)
    parser.add_argument("--output", type=str, default=None, help="Output file name")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--root_path", type=str, default="/home/jonny/Bioinformatics_ExternalData/OGB/")
    parser.add_argument(
        "--target-node-type",
        type=str,
        default="paper",
        help="Node types that predictions are made for",
    )
    args = parser.parse_args()
    print(args)

    random.seed(int(time.time() * 1e6))

    if args.dataset == "mag":
        from ogb.nodeproppred import DglNodePropPredDataset

        home_dir = os.getenv("HOME")
        dataset = DglNodePropPredDataset(
            name="ogbn-mag", root=args.root_path if "root_path" in args else os.path.join(home_dir, ".ogb", "dataset")
        )
        g = dataset[0][0].metagraph()
    elif args.dataset == "acm":
        import sys

        sys.path.append("..")
        from data import load_acm_raw

        dataset = load_acm_raw()
        g = dataset[0].metagraph()
    elif args.dataset.startswith("oag"):
        import pickle
        import dgl

        if args.dataset == "oag_L1":
            graph_file = "../oag_dataset/graph_L1.pk"
        elif args.dataset == "oag_venue":
            graph_file = "../oag_dataset/graph_venue.pk"
        else:
            assert 0
        with open(graph_file, "rb") as f:
            dataset = pickle.load(f)
        g = dgl.heterograph(dataset["edges"]).metagraph()
    else:
        print(f"Dataset {args.dataset} not supported")
        exit(-1)

    subsets = sample_relation_subsets(g, args)

    if args.output is None:
        args.output = "{}_{}_rand_subsets".format(args.num_subsets, args.dataset)
    assert not os.path.exists(args.output)

    with open(args.output, "w") as f:
        for relation in subsets:
            etypes = []

            # only save subsets that touches target node type
            target_touched = False
            for u, v, e in relation:
                etypes.append(e)
                if u == args.target_node_type or v == args.target_node_type:
                    target_touched = True
            print(etypes, target_touched and "touched" or "not touched")
            if target_touched:
                f.write(",".join(etypes) + "\n")
