from argparse import Namespace
from collections.abc import Iterable

import dask.dataframe as dd

from moge.model.dgl.DeepGraphGO import load_dgl_graph, DeepGraphGO


def build_deepgraphgo_model(hparams, base_path="../DeepGraphGO", ):
    if isinstance(hparams.pred_ntypes, str):
        assert len(hparams.pred_ntypes)
        pred_ntypes = hparams.pred_ntypes.split(" ")
    elif isinstance(hparams.pred_ntypes, Iterable):
        pred_ntypes = hparams.pred_ntypes
    else:
        raise Exception("Must provide `hparams.pred_ntypes` as a space-delimited string")

    pred_ntype = pred_ntypes[0]
    namespace = {'molecular_function': 'mf', 'biological_process': 'bp', 'cellular_component': 'cc',
                 'mf': 'mf', 'bp': 'bp', 'cc': 'cc'}[pred_ntype]

    # Proteins list
    df = dd.read_table(f'{base_path}/data/*_*_go.txt',
                       names=['pid', "go_id", 'namespace', 'species_id'],
                       usecols=['pid', 'species_id'])
    groupby = df.groupby('pid')
    proteins = groupby['species_id'].first().to_frame().compute()

    if hparams.dataset == 'HUMAN_MOUSE':
        subset_pid = proteins.query(f'species_id in [9606, 10090]').index
    elif hparams.dataset == 'MULTISPECIES':
        subset_pid = None
    else:
        subset_pid = None

    # Set path to the MultiLabelBinarizer which set the classes

    mlb_path = getattr(hparams, 'mlb_path', f'{base_path}/data/{namespace}_go.mlb')

    # Hparams
    data_cnf = {'mlb': mlb_path,
                'model_path': 'models',
                'name': namespace,
                'protein_data': proteins,
                'network': {'blastdb': f'{base_path}/data/ppi_blastdb',
                            'dgl': f'{base_path}/data/ppi_dgl_top_100',
                            'feature': f'{base_path}/data/ppi_interpro.npz',
                            'pid_list': f'{base_path}/data/ppi_pid_list.txt',
                            'weight_mat': f'{base_path}/data/ppi_mat.npz'},
                'results': 'results',
                'test': {'fasta_file': f'{base_path}/data/{namespace}_test.fasta',
                         'name': 'test',
                         'pid_go_file': f'{base_path}/data/{namespace}_test_go.txt',
                         'pid_list_file': f'{base_path}/data/{namespace}_test_pid_list.txt'},
                'train': {'fasta_file': f'{base_path}/data/{namespace}_train.fasta',
                          'name': 'train',
                          'pid_go_file': f'{base_path}/data/{namespace}_train_go.txt',
                          'pid_list_file': f'{base_path}/data/{namespace}_train_pid_list.txt'},
                'valid': {'fasta_file': f'{base_path}/data/{namespace}_valid.fasta',
                          'name': 'valid',
                          'pid_go_file': f'{base_path}/data/{namespace}_valid_go.txt',
                          'pid_list_file': f'{base_path}/data/{namespace}_valid_pid_list.txt'}}

    batch_size = 120

    model_cnf = {'model': {'hidden_size': 512, 'num_gcn': 2},
                 'name': 'DeepGraphGO',
                 'test': {'batch_size': batch_size},
                 'train': {'batch_size': batch_size, 'epochs_num': 10}}

    # Load dataset
    dgl_graph, node_feats, net_pid_list, train_idx, valid_idx, test_idx = load_dgl_graph(
        data_cnf, model_cnf,
        subset_pid=subset_pid
    )

    hparams = Namespace(**(model_cnf["model"] | model_cnf["train"]),
                        namespace=namespace,
                        nodes=net_pid_list,
                        protein_data=data_cnf['protein_data'],
                        input_size=node_feats.shape[1],
                        n_classes=dgl_graph.ndata["label"].size(1),
                        dropout=0.5,
                        residual=True,
                        training_idx=train_idx,
                        validation_idx=valid_idx,
                        testing_idx=test_idx,
                        lr=1e-3,
                        )
    # load model
    model = DeepGraphGO(hparams,
                        model_path=model_cnf["model"]["model_path"],
                        dgl_graph=dgl_graph,
                        node_feats=node_feats,
                        metrics=["aupr", "fmax"],
                        )

    return model
