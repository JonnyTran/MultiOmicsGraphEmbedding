import copy
import logging
import pickle
import random
import sys
from argparse import ArgumentParser

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)

sys.path.insert(0, "../MultiOmicsGraphEmbedding/")

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping

from moge.generator.multiplex import MultiplexGenerator
from moge.module.trainer import ModelTrainer
from moge.module.multiplex import MultiplexEmbedder
from moge.module.utils import get_multiplex_collate_fn

DATASET = '../MultiOmicsGraphEmbedding/data/proteinatlas_biogrid_multi_network.pickle'
HIER_TAXONOMY_FILE = "../MultiOmicsGraphEmbedding/data/go_term.taxonomy"

def train(hparams):
    with open(DATASET, 'rb') as file:
        network = pickle.load(file)

    MAX_EPOCHS = 30
    min_count = 100

    if hparams.__dict__["encoder.Protein_seqs"] == "Albert":
        hparams.batch_size = 100
        hparams.max_length = 700
    hparams = parse_hparams(hparams)
    print(hparams)

    batch_size = hparams.batch_size
    max_length = hparams.max_length
    n_steps = int(200000 / batch_size)

    variables = []
    targets = ['go_id']
    network.process_feature_tranformer(filter_label=targets[0], delimiter="\||, ", min_count=min_count, verbose=False)
    classes = network.feature_transformer[targets[0]].classes_
    n_classes = len(classes)
    seed = random.randint(0, 1000)

    split_idx = 0
    generator_train = network.get_train_generator(
        MultiplexGenerator, split_idx=split_idx, variables=variables, targets=targets,
        traversal=hparams.traversal, batch_size=batch_size,
        sampling=hparams.sampling, n_steps=n_steps,
        method="GAT", adj_output="coo",
        maxlen=max_length, padding='post', truncating='random',
        seed=seed, verbose=True)

    generator_test = network.get_test_generator(
        MultiplexGenerator, split_idx=split_idx, variables=variables, targets=targets,
        traversal='all_slices',
        batch_size=int(len(network.testing.node_list) * 0.25),
        sampling="cycle", n_steps=1,
        method="GAT", adj_output="coo",
        maxlen=max_length, padding='post', truncating='post',
        seed=seed, verbose=True)

    train_dataloader = torch.utils.data.DataLoader(
        generator_train,
        batch_size=4,
        num_workers=18,
        collate_fn=get_multiplex_collate_fn(node_types=list(hparams.encoder.keys()),
                                            layers=list(hparams.embedder.keys()))
    )

    test_dataloader = torch.utils.data.DataLoader(
        generator_test,
        batch_size=4,
        num_workers=4,
        collate_fn=get_multiplex_collate_fn(node_types=list(hparams.encoder.keys()),
                                            layers=list(hparams.embedder.keys()))
    )

    vocab = generator_train.tokenizer[network.modalities[0]].word_index

    hparams.n_classes = n_classes
    hparams.classes = classes
    hparams.hierar_taxonomy_file = HIER_TAXONOMY_FILE
    hparams.vocab_size = len(vocab)

    logger = WandbLogger()
    # wandb.init(config=hparams, project="multiplex-rna-embedding")

    eec = MultiplexEmbedder(hparams)
    model = ModelTrainer(eec)

    trainer = pl.Trainer(
        distributed_backend='dp',
        gpus=4,
        # auto_lr_find=True,
        logger=logger,
        early_stop_callback=EarlyStopping(monitor='val_loss', patience=3),
        min_epochs=3, max_epochs=MAX_EPOCHS,
        weights_summary='top',
        # amp_level='O1', precision=16,
    )
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=test_dataloader)


def parse_hparams(hparams):
    hparams_dict = copy.copy(hparams.__dict__)
    for key in hparams_dict.keys():
        if "." in key:
            name, value = key.split(".")
            if "seqs" not in value:
                value = value.replace("_", "-")
            if name not in hparams:
                hparams.__setattr__(name, {})
            hparams.__dict__[name][value] = hparams.__dict__[key]
    return hparams


if __name__ == "__main__":
    parser = ArgumentParser()
    # parametrize the network
    parser.add_argument('--encoder.Protein_seqs', type=str, default="ConvLSTM")
    parser.add_argument('--encoding_dim', type=int, default=128)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--word_embedding_size', type=int, default=19)
    parser.add_argument('--max_length', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--traversal', type=str, default="bfs")
    parser.add_argument('--sampling', type=str, default="cycle")

    parser.add_argument('--num_hidden_layers', type=int, default=1)
    parser.add_argument('--num_hidden_groups', type=int, default=1)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.2)
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0.2)
    parser.add_argument('--num_attention_heads', type=int, default=4)
    parser.add_argument('--intermediate_size', type=int, default=128)

    parser.add_argument('--nb_conv1_filters', type=int, default=192)
    parser.add_argument('--nb_conv1_kernel_size', type=int, default=10)
    parser.add_argument('--nb_conv1_dropout', type=float, default=0.2)
    parser.add_argument('--nb_conv1_batchnorm', type=bool, default=True)
    parser.add_argument('--nb_conv2_filters', type=int, default=128)
    parser.add_argument('--nb_conv2_kernel_size', type=int, default=3)
    parser.add_argument('--nb_conv2_batchnorm', type=bool, default=True)
    parser.add_argument('--nb_max_pool_size', type=int, default=13)
    parser.add_argument('--nb_lstm_units', type=int, default=100)
    parser.add_argument('--nb_lstm_bidirectional', type=bool, default=True)
    parser.add_argument('--nb_lstm_hidden_dropout', type=float, default=0.0)
    parser.add_argument('--nb_lstm_layernorm', type=bool, default=True)

    parser.add_argument('--embedder.Protein_Protein_physical', type=str, default="GraphSAGE")
    parser.add_argument('--embedder.Protein_Protein_genetic', type=str, default="GraphSAGE")
    parser.add_argument('--embedder.Protein_Protein_correlation', type=str, default="GraphSAGE")
    parser.add_argument('--nb_attn_heads', type=int, default=2)
    parser.add_argument('--nb_attn_dropout', type=float, default=0.5)

    parser.add_argument('--multiplex_embedder', type=str, default="MultiplexNodeEmbedding")
    parser.add_argument('--multiplex_hidden_dim', type=int, default=512)
    parser.add_argument('--multiplex_attn_dropout', type=float, default=0.2)

    parser.add_argument('--classifier', type=str, default="Dense")
    parser.add_argument('--nb_cls_dense_size', type=int, default=1536)
    parser.add_argument('--nb_cls_dropout', type=float, default=0.2)
    parser.add_argument('--classes_min_count', type=int, default=0)

    parser.add_argument('--use_hierar', type=bool, default=False)
    parser.add_argument('--hierar_penalty', type=float, default=1e-6)

    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--loss_type', type=str, default="BCE_WITH_LOGITS")
    parser.add_argument('--optimizer', type=str, default="adam")

    # add all the available options to the trainer
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    train(args)
