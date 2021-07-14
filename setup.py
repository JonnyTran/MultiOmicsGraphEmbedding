from setuptools import setup, find_packages

requirements = [
    'numpy', 'pandas', 'cmake', 'networkx>=2.1', 'dask', 'biopython', 'bioservices', 'plotly', 'python-igraph',
    'openomics', "tensorflow",
    'chart-studio', "fa2", "scikit-multilearn", "MulticoreTSNE", "gseapy", "focal-loss", "obonet", "wandb",
    "pytorch-lightning", "pytorch_ignite", "ogb"
]

setup(
    name='MultiOmicsGraphEmbedding',
    version='0.2',
    packages=find_packages("moge",
                           include=['moge', 'moge.module', 'moge.network', 'moge.evaluation', 'moge.visualization',
                                    'moge.data'],
                           exclude=["/data*", ]),
    install_requires=requirements,
    url='',
    license='',
    author='JonnyTran',
    author_email='nhat.tran@mavs.uta.edu',
    description=''
)
