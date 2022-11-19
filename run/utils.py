import os
from argparse import ArgumentParser, Namespace
from pprint import pprint
from typing import Union, List

import dgl
import dill
import pynvml
import torch
import yaml
from logzero import logger

from moge.dataset.PyG.node_generator import HeteroNeighborGenerator
from moge.dataset.dgl.node_generator import DGLNodeGenerator
from moge.model.utils import preprocess_input


def parse_yaml_config(parser: ArgumentParser) -> Namespace:
    """

    Args:
        parser ():

    Returns:

    """
    args = parser.parse_args()
    # yaml priority is higher than args
    if isinstance(getattr(args, 'config', None), str) and os.path.exists(args.config):
        opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
        args_dict = args.__dict__

        opt = {k: v for k, v in opt.items if k not in args}
        args_dict.update(opt)
        args = Namespace(**args_dict)

        print("Configs:")
        pprint(opt)
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

    logger.info(f"Adjusted batch_size to", batch_size)

    return int(batch_size)


def select_empty_gpus(num_gpus=1) -> List[int]:
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()

    avail_device = []
    for i in range(deviceCount):
        device = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(device)

        avail_device.append((info.free / info.total, i))

    best_gpu = max(avail_device)[1]
    return [best_gpu]


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
            logger.info(f"Use original features (not embeddings) for node type: {ntype}")
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
