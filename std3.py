import pandas as pd
from numpy import isscalar as np_is_scalar
import numpy as np

from scipy.sparse import hstack
from scipy.sparse import csr_matrix


class Standardization3:
    """
    Standard standardization model for causal inference.
    Learns a model that takes into account the treatment assignment, and later,
    this value can be intervened, changing
    the predicted outcome.
    """

    def __init__(self, learner):
        """
        Args:
            learner: Initialized sklearn model.
        """
        self.learner = learner

    def get_learner(self):
        return self.learner

    def get_method(self):
        return "Standardization"

    def get_learner_string(self):
        return type(self.learner).__name__

    def fit(self, X, a, y):
        if self.get_learner_string() == "GAM":
            new_X = pd.concat([X, a], axis=1)

        else:
            new_a = csr_matrix(a).reshape((X.shape[0], 1))
            new_X = hstack((X, new_a))
        self.learner.fit(new_X, y)
        return self

    def estimate_population_outcome(self, X, a):
        a0_old = pd.Series(0, index=np.arange(len(a)))
        a1_old = pd.Series(1, index=np.arange(len(a)))

        if self.get_learner_string() == "GAM":
            a0 = a0_old
            a1 = a1_old

        else:
            a0 = csr_matrix(a0_old).reshape((a.shape[0], 1))
            a1 = csr_matrix(a1_old).reshape((a.shape[0], 1))

        if self.get_learner_string() == "GAM":
            data0 = pd.DataFrame(X.copy())
            data0["a"] = a0

            data1 = pd.DataFrame(X.copy())
            data1["a"] = a1

        else:
            data0 = hstack((X.copy(), a0))
            data1 = hstack((X.copy(), a1))

        pred_y0 = self.learner.predict(data0)
        std_outcome_est0 = pred_y0.mean()

        pred_y1 = self.learner.predict(data1)
        std_outcome_est1 = pred_y1.mean()
        return std_outcome_est0, std_outcome_est1

    CALCULATE_EFFECT = {
        "diff": lambda x, y: x - y,
        "ratio": lambda x, y: x / y,
        "or": lambda x, y: (x / (1 - x)) / (y / (1 - y)),
    }

    def estimate_effect(self, outcome_1, outcome_2, effect_types):
        results = {}
        for effect_type in effect_types:
            effect = self.CALCULATE_EFFECT[effect_type](outcome_1, outcome_2)
            results[effect_type] = effect
        # Format results in pandas array:
        results = (
            pd.Series(results)
            if np_is_scalar(outcome_1)
            else pd.concat(results, axis="columns", names=["effect_type"])
        )
        return results
