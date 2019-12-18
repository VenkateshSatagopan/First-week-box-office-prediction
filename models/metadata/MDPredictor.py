import numpy as np

from .MDPred1 import MDPred1
from .MDPred2 import MDPred2


class MDPredictor(object):
    def __init__(self):
        model1 = MDPred1()
        model2 = MDPred2()
        self.models = [model1, model2]

    def __call__(self, metadata_dataframe):
        return self.predict(metadata_dataframe)

    def predict(self, input_vector):
        result = []
        for m in self.models:
            pred = m(input_vector)
            result.append(pred)
        return result
