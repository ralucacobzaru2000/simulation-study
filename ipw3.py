import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from numpy import isscalar as np_is_scalar
from sklearn.calibration import calibration_curve


class IPW3:
    """
    Causal model implementing inverse probability (propensity score) weighting.
    w_i = 1 / Pr[A=a_i|Xi]
    """

    def __init__(self, learner):
        """
        learner: initialized sklearn model
        """
        self.learner = learner

    def get_learner(self):
        return self.learner

    def get_method(self):
        return "IPW"

    def get_learner_string(self):
        return type(self.learner).__name__

    def fit(self, X, a):
        """
        X: matrix of covariates
        a: treatment values
        """
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            X = X.toarray()
        self.learner.fit(X, a)
        return self

    def estimate_population_outcome(self, X, a, y, showplot=False, truncate=None):
        """
        Converts everything to pandas
        """
        # get probability that treatment = 1
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            X = X.toarray()
        treatment_probs = self.learner.predict_proba(X)

        if self.get_learner_string() == "LogisticGAM":
            prob = pd.DataFrame(data=treatment_probs, columns=["treatment_probs"])
        else:
            prob = pd.DataFrame(treatment_probs[:, -1])

        prob = pd.concat([prob, a, y], axis=1, ignore_index=True)
        prob.columns = ["treatment_probs", "a", "y"]

        prob["treatprob_0"] = 1 - prob["treatment_probs"]

        check_weights = (1 / prob["treatment_probs"]).append(
            1 / (1 - prob["treatment_probs"])
        )

        if truncate is not None:
            # truncate is percentage float
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

        est1 = prob["y"] * prob["a"] / prob["treatment_probs"]
        est0 = prob["y"] * (1 - prob["a"]) / prob["treatprob_0"]

        ipw_outcome_est0 = est0.mean()
        ipw_outcome_est0_df = pd.Series(ipw_outcome_est0)

        ipw_outcome_est1 = est1.mean()
        ipw_outcome_est1_df = pd.Series(ipw_outcome_est1)

        if showplot:
            # log reg predict proba
            logreg_model = IPW3(LogisticRegression())
            logreg_model.fit(X, a)
            logprobs = logreg_model.learner.predict_proba(X)
            logprob = pd.DataFrame(logprobs[:, -1])
            logprob.columns = ["logprob"]
            logreg_y, logreg_x = calibration_curve(y_true=a, y_prob=logprob["logprob"])

            fig, ax = plt.subplots()

            plot_y, plot_x = calibration_curve(y_true=a, y_prob=prob["treatment_probs"])
            plt.plot(logreg_x, logreg_y, marker="o", label="logreg")
            plt.plot(plot_x, plot_y, marker="o", label=self.get_learner_string())

            ax.set_xlabel("predicted probability")
            ax.set_ylabel("true probability")
            plt.legend()
            plt.show()

        return ipw_outcome_est0_df, ipw_outcome_est1_df

    def plot_calibration_curve(self, X, a, y):
        # logistic reg w/ no calibration as baseline
        # lr = LogisticRegression()
        # fig = plt.figure
        treatment_probs = self.learner.predict_proba(X)
        prob = pd.DataFrame(treatment_probs[:, -1])

        plt.scatter(np.mean(prob))

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
