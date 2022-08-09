from setuptools import setup, find_packages

requirements = [
    'numpy', 'pandas', 'networkx>=2.1', "boto3",
    'dask', 'plotly', 'python-igraph', "colorhash",
    'biopython', 'bioservices', "gseapy", "obonet", 'iterative-stratification', 'scikit-multilearn',
    "wandb", "pytorch-lightning", "pytorch-ignite", "ogb", "torchmetrics", "torchtext", "fairscale",
    "logzero"
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
