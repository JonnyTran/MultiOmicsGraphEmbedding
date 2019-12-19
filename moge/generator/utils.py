from typing import Generator

import numpy as np


class EdgelistSampler(Generator):
    def __init__(self, edgelist: list):
        """
        This class is used to perform sampling without replacement from a node's edgelist
        :param edgelist:
        """
        assert type(edgelist) is list
        self.edgelist = edgelist
        self.sampled_idx = list(np.random.choice(range(len(self.edgelist)), size=len(self.edgelist), replace=False))

    def send(self, ignored_arg=None):
        while True:
            if len(self.sampled_idx) > 0:
                return self.edgelist[self.sampled_idx.pop()]
            else:
                self.sampled_idx = list(
                    np.random.choice(range(len(self.edgelist)), size=len(self.edgelist), replace=False))

    def throw(self, type=None, value=None, traceback=None):
        raise StopIteration

    def append(self, item):
        self.edgelist.append(item)

    def __len__(self):
        return len(self.edgelist)
