import pandas as pd
from numpy import isscalar as np_is_scalar
import numpy as np
import utilities
from scipy.sparse import hstack
from scipy.sparse import csr_matrix


class DoublyRobust:
    def __init__(self, outcome_model, weight_model):
        """
        Args:
            outcome_model(IndividualOutcomeEstimator): A causal model that estimate on individuals level
                                                      (e.g. Standardization).
            weight_model (WeightEstimator): A causal model for weighting individuals (e.g. IPW).
        """
        self.outcome_model = outcome_model
        self.weight_model = weight_model

    def fit(self, X, a, y):
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            X = X.toarray()
        X1_pd = pd.DataFrame(X)
        X1_pd.to_csv("X1_dr_fit.csv")
        self.weight_model.learner.fit(X, a)
        if (
            self.get_out_model().get_learner_string() == "GAM"
            or self.get_out_model().get_learner_string() == "LogisticGAM"
            or self.get_treat_model().get_learner_string() == "LogisticGAM"
            or self.get_treat_model().get_learner_string() == "SuperLearner"
            or self.get_out_model().get_learner_string() == "SuperLearner"
        ):
            new_X = pd.concat([X, a], axis=1)

        else:
            new_a = csr_matrix(a).reshape((X.shape[0], 1))
            new_X = hstack((X, new_a))

        if not isinstance(new_X, (np.ndarray, pd.DataFrame)):
            new_X = new_X.toarray()
        self.outcome_model.learner.fit(new_X, y)
        return self

    def get_method(self):
        return "Double Robust"

    def get_out_model(self):
        return self.outcome_model

    def get_treat_model(self):
        return self.weight_model

    def get_out_model_string(self):
        return type(self.outcome_model).__name__

    def get_treat_model_string(self):
        return type(self.weight_model).__name__

    def get_learner_string(self):
        return (
            self.outcome_model.get_learner_string(),
            self.weight_model.get_learner_string(),
        )

    def estimate_population_outcome(self, X, a, y, truncate=None, sandwich=False):
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            X = X.toarray()
        if type(self.weight_model.learner) == utilities.SuperLearner:
            treatment_probs = self.weight_model.learner.predict(X)
        else:
            treatment_probs = self.weight_model.learner.predict_proba(X)

        a0_old = pd.Series(0, index=np.arange(len(a)))
        a1_old = pd.Series(1, index=np.arange(len(a)))

        if (
            self.get_out_model().get_learner_string() == "GAM"
            or self.get_out_model().get_learner_string() == "LogisticGAM"
            or self.get_treat_model().get_learner_string() == "LogisticGAM"
            or self.get_treat_model().get_learner_string() == "SuperLearner"
            or self.get_out_model().get_learner_string() == "SuperLearner"
        ):
            prob = pd.DataFrame(data=treatment_probs, columns=["treatment_probs"])
            # if GAM or LogisticGAM, return np.array
            a0 = a0_old
            a1 = a1_old
        else:
            prob = pd.DataFrame(treatment_probs[:, -1])
            prob.columns = ["treatment_probs"]
            a0 = csr_matrix(a0_old).reshape((a.shape[0], 1))
            a1 = csr_matrix(a1_old).reshape((a.shape[0], 1))

        prob["a"] = a.copy()
        prob["y"] = y.copy()

        prob["treatprob_0"] = 1 - prob["treatment_probs"]

        # simul paper trim
        if truncate is not None:

            change_indices = [
                i for i, value in enumerate(prob["treatprob_0"]) if value < truncate
            ]
            for index in change_indices:
                prob["treatprob_0"].at[index] = truncate

            change_indices_1 = [
                i
                for i, value_1 in enumerate(prob["treatment_probs"])
                if value_1 < truncate
            ]
            for index in change_indices_1:
                prob["treatment_probs"].at[index] = truncate

            change_indices = [
                i for i, value in enumerate(prob["treatprob_0"]) if value > 1 - truncate
            ]
            for index in change_indices:
                prob["treatprob_0"].at[index] = 1 - truncate

            change_indices_1 = [
                i
                for i, value_1 in enumerate(prob["treatment_probs"])
                if value_1 > 1 - truncate
            ]
            for index in change_indices_1:
                prob["treatment_probs"].at[index] = 1 - truncate

        if (
            self.get_out_model().get_learner_string() == "GAM"
            or self.get_out_model().get_learner_string() == "LogisticGAM"
            or self.get_treat_model().get_learner_string() == "LogisticGAM"
            or self.get_out_model().get_learner_string() == "SuperLearner"
        ):
            # use this, working older
            data0 = pd.DataFrame(X.copy())
            data0["a"] = a0

            data1 = pd.DataFrame(X.copy())
            data1["a"] = a1

        else:
            data0 = hstack((X.copy(), a0))
            data1 = hstack((X.copy(), a1))

        if not isinstance(data0, (np.ndarray, pd.DataFrame)):
            data0 = data0.toarray()
        pred_y0 = self.outcome_model.learner.predict(data0)

        est0 = prob["y"] * (1 - prob["a"]) / (prob["treatprob_0"]) + (
            prob["a"] - prob["treatment_probs"]
        ) * pred_y0 / (prob["treatprob_0"])

        # from paper
        # est0 = prob['y'] * (1 - prob['a']) - (prob['a'] - prob['treatment_probs']) * pred_y0 * all_weights
        # from slides
        # est0 = (prob['y'] * (1 - prob['a']) + (prob['a'] - prob['treatment_probs']) * pred_y0) * all_weights
        dr_outcome_est0 = est0.mean()
        dr_outcome_est0_df = pd.Series(dr_outcome_est0)

        if not isinstance(data1, (np.ndarray, pd.DataFrame)):
            data1 = data1.toarray()
        pred_y1 = self.outcome_model.learner.predict(data1)

        est1 = (prob["y"] * prob["a"]) / prob["treatment_probs"] - (
            (prob["a"] - prob["treatment_probs"]) * pred_y1 / prob["treatment_probs"]
        )

        dr_outcome_est1 = est1.mean()
        dr_outcome_est1_df = pd.Series(dr_outcome_est1)

        if sandwich:
            diff = dr_outcome_est1_df - dr_outcome_est0_df
            sand = est1 - est0 - float(diff)
            n = X.shape[0]
            std_error = pd.Series(np.sqrt(1 / n ** 2 * np.sum(sand ** 2)))
            return dr_outcome_est0_df, dr_outcome_est1_df, diff, std_error

        return dr_outcome_est0_df, dr_outcome_est1_df

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
            else pd.concat(results, axis="columns", names=["effect_type"], sort=True)
        )
        return results
