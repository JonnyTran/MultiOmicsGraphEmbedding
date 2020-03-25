import pandas as pd


def edges_dataframe(network, data=False):
    edges_list = pd.DataFrame.from_dict(network.edges(data=data), orient='columns', )
    edges_list.rename(columns={0: "source", 1: "target"}, inplace=True)
    return edges_list


def get_rename_dict(dataframe, alias_col_name):
    dataframe = dataframe[dataframe[alias_col_name].notnull()]
    b = pd.DataFrame(dataframe[alias_col_name].str.split('|').tolist(), index=dataframe.index).stack()
    b = b.reset_index(level=0)
    b.columns = ['index', 'alias']
    b.index = b["alias"]
    b = b.reindex()
    return pd.Series(b["index"]).to_dict()
