from setuptools import setup

requirements = [
    'numpy', 'pandas', 'networkx>=2.1', 'dask', 'biopython', 'bioservices', 'plotly', 'python-igraph', 'chart-studio',
    "fa2", "scikit-multilearn", "MulticoreTSNE", "gseapy",
]

setup(
    name='MultiOmicsGraphEmbedding',
    version='0.1',
    packages=['moge', 'moge.model', 'moge.network', 'moge.evaluation', 'moge.visualization', 'moge.generator'],
    install_requires=requirements,
    url='',
    license='',
    author='JonnyTran',
    author_email='nhat.tran@mavs.uta.edu',
    description=''
)
