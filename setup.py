from setuptools import setup, find_packages

requirements = [
    'numpy', 'pandas', 'cmake', 'networkx>=2.1', 'dask', 'biopython', 'bioservices', 'plotly', 'python-igraph',
    'chart-studio',
    "fa2", "scikit-multilearn", "MulticoreTSNE", "gseapy", "focal-loss", "obonet", "wandb"
]

setup(
    name='MultiOmicsGraphEmbedding',
    version='0.2',
    packages=find_packages("moge",
                           include=['moge', 'moge.model', 'moge.network', 'moge.evaluation', 'moge.visualization',
                                    'moge.generator'],
                           exclude=["*data*", "moge.data*"]),
    install_requires=requirements,
    url='',
    license='',
    author='JonnyTran',
    author_email='nhat.tran@mavs.uta.edu',
    description=''
)
