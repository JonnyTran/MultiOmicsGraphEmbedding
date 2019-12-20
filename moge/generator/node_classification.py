import pandas as pd
from .data_generator import DataGenerator
import pandas as pd

from .data_generator import DataGenerator


class ClassificationGenerator(DataGenerator):
    def __init__(self, variables, targets, **kwargs):
        self.variables = variables
        self.targets = targets
        super(ClassificationGenerator, self).__init__(**kwargs)

    def get_targets(self, sampled_nodes):
        y = pd.get_dummies(self.annotations.loc[sampled_nodes], columns=self.targets,
                           dummy_na=False).to_numpy()
        return y

    def get_variables(self, sampled_nodes):
        X = {}
        for variable in self.variables:
            X[variable] = pd.get_dummies(self.annotations.loc[sampled_nodes], columns=[variable, ],
                                         dummy_na=False).to_numpy()

        return X
