import logging
import os
from argparse import ArgumentParser, Namespace
from pprint import pprint
from typing import Union

import dgl
import dill
import torch
import yaml

from moge.dataset.PyG.node_generator import HeteroNeighborGenerator
from moge.dataset.dgl.node_generator import DGLNodeGenerator
from moge.model.utils import preprocess_input


def parse_yaml_config(parser: ArgumentParser) -> Namespace:
    parser.add_argument('-y', '--config', help="configuration file *.yml", type=str, required=False)
    args = parser.parse_args()
    # yaml priority is higher than args
    if isinstance(args.config, str) and os.path.exists(args.config):
        opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
        args_dict = args.__dict__
        args_dict.update(opt)
        args = Namespace(**args_dict)

        print("Configs:")
        pprint(args.__dict__)
        print()

    return args


def adjust_batch_size(hparams):
    batch_size = hparams.batch_size
    if batch_size < 0: return batch_size

    if hparams.n_neighbors > 256:
        batch_size = batch_size // (hparams.n_neighbors // 128)
    if hparams.embedding_dim > 128:
        batch_size = batch_size // (hparams.embedding_dim // 128)
    if hparams.n_layers > 2:
        batch_size = batch_size // (hparams.n_layers - 1)
    if hparams.neg_sampling_ratio != 1000:
        batch_size = batch_size // (hparams.neg_sampling_ratio / 1000)

    print(f"Adjusted batch_size to", batch_size)

    return int(batch_size)


def select_empty_gpu():
    gpu_mem_free = {i: torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())}
    best_gpu = max(gpu_mem_free, key=gpu_mem_free.get)

    del gpu_mem_free
    torch.cuda.empty_cache()
    return best_gpu


def add_node_embeddings(dataset: Union[HeteroNeighborGenerator, DGLNodeGenerator], path: str, skip_ntype: str = None,
                        args: Namespace = None):
    node_emb = {}
    if os.path.exists(path) and os.path.isdir(path):
        for file in os.listdir(path):
            ntype = file.split(".")[0]
            ndata = torch.load(os.path.join(path, file)).float()

            node_emb[ntype] = ndata

    elif os.path.exists(path) and os.path.isfile(path):
        features = dill.load(open(path, 'rb'))  # Assumes .pk file

        for ntype, ndata in preprocess_input(features, device="cpu", dtype=torch.float).items():
            node_emb[ntype] = ndata
    else:
        print(f"Failed to import embeddings from {path}")

    for ntype, ndata in node_emb.items():
        if skip_ntype == ntype:
            logging.info(f"Use original features (not embeddings) for node type: {ntype}")
            continue

        if "freeze_embeddings" in args and args.freeze_embeddings == False:
            print("got here")
            if "node_emb_init" not in args:
                args.node_emb_init = {}

            args.node_emb_init[ntype] = ndata

        elif isinstance(dataset.G, dgl.DGLHeteroGraph):
            dataset.G.nodes[ntype].data["feat"] = ndata

        elif isinstance(dataset.G, HeteroNeighborGenerator):
            dataset.G.x_dict[ntype] = ndata
        else:
            raise Exception(f"Cannot recognize type of {dataset.G}")

        print(f"Loaded embeddings for {ntype}: {ndata.shape}")
