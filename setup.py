import pandas as pd
from setuptools import setup, find_packages

requirements = pd.read_table("./requirements.txt", header=None)[0].tolist()
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
