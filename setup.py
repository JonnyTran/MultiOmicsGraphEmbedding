from setuptools import setup, find_packages

requirements = [
    'numpy', 'pandas', 'cmake', 'networkx>=2.1', 'dask', 'biopython', 'bioservices', 'plotly', 'python-igraph',
    "colorhash", "scikit-multilearn", "gseapy",
    "obonet", "wandb",
    "pytorch-lightning", "pytorch-ignite", "ogb", "torchmetrics",
]

setup(
    name='MultiOmicsGraphEmbedding',
    version='0.3',
    packages=find_packages("moge",
                           include=['moge'],
                           exclude=["/data/", ]),
    install_requires=requirements,
    url='https://github.com/JonnyTran/MultiOmicsGraphEmbedding',
    license='MIT',
    author='JonnyTran',
    author_email='nhat.tran@mavs.uta.edu',
    description='',
)
