import time
import warnings
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from config_constants import JsonConfigConstants as JCst
from double_robust2 import DoublyRobust
from ipw3 import IPW3
from std3 import Standardization3
from utilities import Utilities
from scipy.sparse import csr_matrix

# comment in if running main simulations with dgp; out if running extra_sim simulations
# from extra_sim import PatientDGP

warnings.filterwarnings("ignore")

MEAN: str = Utilities.MEAN
STDEV: str = Utilities.STDEV
CONF_INT_95: str = Utilities.CONF_INT_95
CONF_95_LOWER: str = Utilities.CONF_95_LOWER
CONF_95_UPPER: str = Utilities.CONF_95_UPPER
CONF_95: float = 1.96
utils = Utilities()


class MasterModel:
    COLUMN_NAMES: Dict[str, Dict[str, str]] = {
        "diff": {
            MEAN: "diff mean",
            STDEV: "diff std",
            CONF_95_LOWER: "diff lower",
            CONF_95_UPPER: "diff upper",
        },
        "ratio": {
            MEAN: "ratio mean",
            STDEV: "ratio std",
            CONF_95_LOWER: "ratio lower",
            CONF_95_UPPER: "ratio upper",
        },
        "or": {
            MEAN: "odds ratio mean",
            STDEV: "odds ratio std",
            CONF_95_LOWER: "odds ratio lower",
            CONF_95_UPPER: "odds ratio upper",
        },
    }

    def __init__(self, config_file: str, sample_size=3000) -> None:
        # map learner str to actual sklearn model
        # map config to actual instances of methods and their learner models
        self.models = []
        input_dict = utils.get_json_data(config_file)

        if input_dict[JCst.ADJ_METHOD][JCst.IPW]:
            learners = input_dict[JCst.LEARNER][JCst.IPW]
            self.models.extend(utils.build_model_list(learners, JCst.IPW)[0])

        if input_dict[JCst.ADJ_METHOD][JCst.STANDARDIZATION]:
            learners = input_dict[JCst.LEARNER][JCst.STANDARDIZATION]
            self.models.extend(
                utils.build_model_list(learners, JCst.STANDARDIZATION)[0]
            )

        if input_dict[JCst.ADJ_METHOD][JCst.D_ROBUST]:
            outcome_model = input_dict[JCst.D_ROBUST_MODEL][JCst.OUTCOME_MODEL]
            outcome_learner = input_dict[JCst.LEARNER][outcome_model[0]]
            out_models = utils.build_model_list(outcome_learner, outcome_model[0])[0]

            treatment_model = input_dict[JCst.D_ROBUST_MODEL][JCst.TREATMENT_MODEL]
            weight_learner = input_dict[JCst.LEARNER][treatment_model[0]]
            wei_models = utils.build_model_list(weight_learner, treatment_model[0])[0]

            for out_model in out_models:
                for wei_model in wei_models:
                    # outcome model is first, then weight model is 2nd
                    model = DoublyRobust(out_model, wei_model)
                    self.models.append(model)

        self.sample_split = input_dict[JCst.D_ROBUST_MODEL][JCst.SAMPLE_SPLIT]
        self.outcome_type = input_dict[JCst.EFF_MEASURE][JCst.OUTCOME_TYPE]
        self.effect_types = input_dict[JCst.EFF_MEASURE][JCst.RELATION]
        self.std_types = input_dict[JCst.ADVANCED][JCst.STD_ERR]
        self.bootstrap = input_dict[JCst.ADVANCED][JCst.STD_ERR][JCst.BOOTSTRAP]
        self.analytic = input_dict[JCst.ADVANCED][JCst.STD_ERR][JCst.ANALYTIC]
        self.learners_IPW = input_dict[JCst.LEARNER][JCst.IPW]
        self.learners_standardization = input_dict[JCst.LEARNER][JCst.STANDARDIZATION]
        self.truncate_percent = input_dict[JCst.ADVANCED][JCst.WEIGHT_STABILIZE][
            JCst.TRUNCATE
        ]

        self.check_decompress = input_dict[JCst.ADVANCED][JCst.DATA_PROCESS][
            JCst.CHECK_DECOMPRESS
        ]
        self.multi_treat = input_dict[JCst.ADVANCED][JCst.DATA_PROCESS][
            JCst.MULTI_TREAT
        ]
        self.multi_treat_iters = input_dict[JCst.ADVANCED][JCst.DATA_PROCESS][
            JCst.MULTI_TREAT_ITERS
        ]
        self.only_sparse_covar = input_dict[JCst.ADVANCED][JCst.DATA_PROCESS][
            JCst.ONLY_SPARSE_COVAR
        ]
        self.bootstrap_seed = input_dict[JCst.ADVANCED][JCst.DATA_PROCESS][
            JCst.BOOTSTRAP_SEED
        ]

    def process_dataset(self, data: pd.DataFrame) -> List[Dict[str, Union[float, str]]]:
        X_list = [column for column in data if column.startswith("X")]
        X = data[X_list]
        a = data["A"]
        y = data["Y"]
        row_list = []

        # handle multiple combos of methods (IPW, std, dr)
        for model in self.models:
            start_time = time.time()

            if type(model) == DoublyRobust and self.analytic:
                X_list = [column for column in data if column.startswith("X")]

                dict1 = {}
                dict1["method"] = model.get_method()
                dict1["std error"] = "analytic"
                dict1["outcome model"] = model.get_out_model_string()
                dict1["treatment model"] = model.get_treat_model_string()
                dict1["out learner"] = model.get_learner_string()[0]
                dict1["treat learner"] = model.get_learner_string()[1]

                if self.sample_split:
                    dict1["sample split"] = "True"
                    # incorporate sample splitting, randomly splits data in 2 equal samples
                    sample_1, sample_2 = np.array_split(data, 2)
                    sample_1.reset_index(inplace=True)
                    sample_2.reset_index(inplace=True)

                    # getting all other columns in X,A,Y for both samples
                    column_list = [column for column in data if column.startswith("X")]
                    X1_sample_old = sample_1[column_list]
                    a1_sample = sample_1["A"]
                    y1_sample = sample_1["Y"]

                    X2_sample_old = sample_2[column_list]
                    a2_sample = sample_2["A"]
                    y2_sample = sample_2["Y"]

                    # sparse format
                    X1_sample = csr_matrix(X1_sample_old, dtype=float)
                    X2_sample = csr_matrix(X2_sample_old, dtype=float)
                    if (
                        model.get_treat_model().get_learner_string() == "LogisticGAM"
                        or model.get_out_model().get_learner_string() == "LogisticGAM"
                        or model.get_out_model().get_learner_string() == "GAM"
                        or model.get_treat_model().get_learner_string()
                        == "SuperLearner"
                        or model.get_out_model().get_learner_string() == "SuperLearner"
                    ):
                        X1_sample = X1_sample_old
                        X2_sample = X2_sample_old

                    model.fit(X1_sample, a1_sample, y1_sample)
                    if len(self.truncate_percent) != 0:
                        (
                            outcome_piece2_0,
                            outcome_piece2_1,
                            diff2,
                            std_error2,
                        ) = model.estimate_population_outcome(
                            X2_sample,
                            a2_sample,
                            y2_sample,
                            truncate=self.truncate_percent[0],
                            sandwich=True,
                        )
                    else:
                        (
                            outcome_piece2_0,
                            outcome_piece2_1,
                            diff2,
                            std_error2,
                        ) = model.estimate_population_outcome(
                            X2_sample, a2_sample, y2_sample, sandwich=True
                        )
                    eff_piece2 = model.estimate_effect(
                        outcome_piece2_1, outcome_piece2_0, self.effect_types
                    )

                    # fit on piece 2, estimate effect on piece 1
                    # if GAM or logistic GAM:
                    model.fit(X2_sample, a2_sample, y2_sample)

                    if len(self.truncate_percent) != 0:
                        (
                            outcome_piece1_0,
                            outcome_piece1_1,
                            diff1,
                            std_error1,
                        ) = model.estimate_population_outcome(
                            X1_sample,
                            a1_sample,
                            y1_sample,
                            truncate=self.truncate_percent[0],
                            sandwich=True,
                        )
                    else:
                        (
                            outcome_piece1_0,
                            outcome_piece1_1,
                            diff1,
                            std_error1,
                        ) = model.estimate_population_outcome(
                            X1_sample, a1_sample, y1_sample, sandwich=True
                        )
                    eff_piece1 = model.estimate_effect(
                        outcome_piece1_1, outcome_piece1_0, self.effect_types
                    )

                    est0 = (outcome_piece2_0 + outcome_piece1_0) / 2
                    est1 = (outcome_piece2_1 + outcome_piece1_1) / 2
                    avg_eff = (eff_piece1 + eff_piece2) / 2
                    std_error = (std_error1 + std_error2) / 2

                else:  # no sample split
                    dict1["sample split"] = "False"
                    model.fit(X, a, y)
                    if len(self.truncate_percent) != 0:
                        (
                            est0,
                            est1,
                            diff,
                            std_error,
                        ) = model.estimate_population_outcome(
                            X, a, y, truncate=self.truncate_percent[0], sandwich=True,
                        )
                    else:
                        (
                            est0,
                            est1,
                            diff,
                            std_error,
                        ) = model.estimate_population_outcome(X, a, y, sandwich=True)
                    avg_eff = model.estimate_effect(est1, est0, self.effect_types)

                if type(model.get_out_model().get_learner()).__name__ in {
                    "StackingRegressor",
                    "VotingRegressor",
                }:
                    ens_learners = model.get_out_model().get_learner().estimators_
                    dict1["out ensemble learners"] = str(
                        [type(learn).__name__ for learn in ens_learners]
                    )

                if type(model.get_treat_model().get_learner()).__name__ in {
                    "StackingClassifier",
                    "VotingClassifier",
                }:
                    ens_learners = model.get_treat_model().get_learner().estimators_
                    dict1["treat ensemble learners"] = str(
                        [type(learn).__name__ for learn in ens_learners]
                    )

                if not isinstance(std_error, float):
                    std_error = std_error[0]

                # treatment = 0
                dict1["a=0 mean"] = est0.mean()
                dict1["a=0 lower"] = est0.mean() - CONF_95 * std_error
                dict1["a=0 upper"] = est0.mean() + CONF_95 * std_error

                # treatment = 1
                dict1["a=1 mean"] = est1.mean()
                dict1["a=1 lower"] = est1.mean() - (CONF_95 * std_error)
                dict1["a=1 upper"] = est1.mean() + (CONF_95 * std_error)

                # effect estimate
                try:
                    eff_column = list(avg_eff.columns)
                except Exception:
                    eff_column = list(avg_eff.index)
                for col_name in eff_column:
                    metrics = utils.compute_metrics(avg_eff[col_name])
                    stats_names = self.COLUMN_NAMES[col_name]
                    dict1[stats_names[MEAN]] = metrics[MEAN]
                    dict1[stats_names[STDEV]] = std_error
                    dict1[stats_names[CONF_95_LOWER]] = (
                        metrics[MEAN] - CONF_95 * std_error
                    )
                    dict1[stats_names[CONF_95_UPPER]] = (
                        metrics[MEAN] + CONF_95 * std_error
                    )

                dict1["time"] = time.time() - start_time
                row_list.append(dict1)

            if self.bootstrap == 1:
                # configure bootstrap
                n_iterations = 100
                n_size = int(len(data))

                stats0 = pd.Series([])
                stats1 = pd.Series([])
                effs = pd.DataFrame([])

                #################################

                if type(model) == DoublyRobust:
                    if self.sample_split == 1:
                        sample_1, sample_2 = np.array_split(data, 2)
                        sample_1.reset_index(inplace=True)
                        sample_2.reset_index(inplace=True)

                        # getting all other columns
                        column_list = [
                            column for column in data if column.startswith("X")
                        ]
                        X1_sample_old = sample_1[column_list]
                        a1_sample = sample_1["A"]
                        y1_sample = sample_1["Y"]

                        X2_sample_old = sample_2[column_list]
                        a2_sample = sample_2["A"]
                        y2_sample = sample_2["Y"]

                        # sparse format
                        X1_sample = csr_matrix(X1_sample_old, dtype=float)
                        X2_sample = csr_matrix(X2_sample_old, dtype=float)

                        # if GAM or logistic GAM:
                        if (
                            model.get_treat_model().get_learner_string()
                            == "LogisticGAM"
                            or model.get_out_model().get_learner_string()
                            == "LogisticGAM"
                            or model.get_out_model().get_learner_string() == "GAM"
                            or model.get_treat_model().get_learner_string()
                            == "SuperLearner"
                            or model.get_out_model().get_learner_string()
                            == "SuperLearner"
                        ):
                            X1_sample = X1_sample_old
                            X2_sample = X2_sample_old

                        # fit on piece 1, estimate effect on piece 2
                        model.fit(X1_sample, a1_sample, y1_sample)

                        if len(self.truncate_percent) != 0:
                            (
                                outcome_piece2_0,
                                outcome_piece2_1,
                            ) = model.estimate_population_outcome(
                                X2_sample,
                                a2_sample,
                                y2_sample,
                                truncate=self.truncate_percent[0],
                            )
                        else:
                            (
                                outcome_piece2_0,
                                outcome_piece2_1,
                            ) = model.estimate_population_outcome(
                                X2_sample, a2_sample, y2_sample
                            )
                        eff_piece2 = model.estimate_effect(
                            outcome_piece2_1, outcome_piece2_0, self.effect_types
                        )

                        # fit on piece 2, estimate effect on piece 1
                        model.fit(X2_sample, a2_sample, y2_sample)

                        if len(self.truncate_percent) != 0:
                            (
                                outcome_piece1_0,
                                outcome_piece1_1,
                            ) = model.estimate_population_outcome(
                                X1_sample,
                                a1_sample,
                                y1_sample,
                                truncate=self.truncate_percent[0],
                            )
                        else:
                            (
                                outcome_piece1_0,
                                outcome_piece1_1,
                            ) = model.estimate_population_outcome(
                                X1_sample, a1_sample, y1_sample
                            )
                        eff_piece1 = model.estimate_effect(
                            outcome_piece1_1, outcome_piece1_0, self.effect_types
                        )

                        est0 = (outcome_piece2_0 + outcome_piece1_0) / 2
                        est1 = (outcome_piece2_1 + outcome_piece1_1) / 2
                        avg_eff = (eff_piece1 + eff_piece2) / 2

                    else:  # no sample split
                        dict1["sample split"] = "False"
                        model.fit(X, a, y)
                        if len(self.truncate_percent) != 0:
                            est0, est1 = model.estimate_population_outcome(
                                X, a, y, truncate=self.truncate_percent[0]
                            )
                        else:
                            est0, est1 = model.estimate_population_outcome(X, a, y)
                        avg_eff = model.estimate_effect(est1, est0, self.effect_types)
                """
                if (
                    model.get_learner_string() == "LogisticGAM"
                    or model.get_learner_string() == "GAM"
                    or model.get_treat_model().get_learner_string()
                    == "SuperLearner"
                    or model.get_out_model().get_learner_string() == "SuperLearner"
                ):
                    X = X_old
                """
                if type(model) == IPW3:
                    model.fit(X, a)
                    self.weight_model = None
                    if len(self.truncate_percent) != 0:
                        # there is percentage specified to truncate weights
                        est0, est1 = model.estimate_population_outcome(
                            X, a, y, truncate=self.truncate_percent[0]
                        )
                    else:
                        est0, est1 = model.estimate_population_outcome(X, a, y)

                    avg_eff = model.estimate_effect(est1, est0, self.effect_types)

                if type(model) == Standardization3:
                    # print(X)
                    # print(type(X))
                    model.fit(X, a, y)
                    est0, est1 = model.estimate_population_outcome(X, a)
                    avg_eff = model.estimate_effect(est1, est0, self.effect_types)

                for i in range(n_iterations):
                    # find estimate outcomes for the size n sample for n_iterations times
                    # sample = df.sample(n=n_size,frac=None,replace=True,weights=None,random_state=None,axis=None)
                    print("Iteration", i)
                    sample = data.sample(
                        n=n_size,
                        frac=None,
                        replace=True,
                        weights=None,
                        random_state=None,
                        axis=None,
                    )
                    sample.reset_index(inplace=True)

                    # getting all other columns
                    column_list = [
                        column for column in sample if column.startswith("X")
                    ]
                    X_sample_old = sample[column_list]
                    a_sample = sample["A"]
                    y_sample = sample["Y"]

                    # sparse format
                    X_sample = csr_matrix(X_sample_old, dtype=float)

                    if (
                        model.get_learner_string() == "LogisticGAM"
                        or model.get_learner_string() == "GAM"
                        # or model.get_treat_model().get_learner_string()
                        # == "SuperLearner"
                        # or model.get_out_model().get_learner_string()
                        # == "SuperLearner"
                    ):
                        X_sample = X_sample_old
                    # ipw fit: only X,a
                    if type(model) == IPW3:
                        model.fit(X_sample, a_sample)

                        if len(self.truncate_percent) != 0:
                            # there is percentage specified to truncate weights
                            est0, est1 = model.estimate_population_outcome(
                                X_sample,
                                a_sample,
                                y_sample,
                                truncate=self.truncate_percent[0],
                            )
                        else:
                            est0, est1 = model.estimate_population_outcome(
                                X_sample, a_sample, y_sample
                            )

                        stats0 = stats0.append(est0, ignore_index=True)
                        stats1 = stats1.append(est1, ignore_index=True)

                    elif type(model) == Standardization3:
                        model.fit(X_sample, a_sample, y_sample)
                        est0, est1 = model.estimate_population_outcome(
                            X_sample, a_sample
                        )

                        stats0 = stats0.append(pd.Series(est0), ignore_index=True)
                        stats1 = stats1.append(pd.Series(est1), ignore_index=True)

                    elif type(model) == DoublyRobust:
                        if self.sample_split == 1:
                            # shuffled = data.sample(frac=1)
                            # sample_1, sample_2 = np.array_split(shuffled, 2)
                            sample_1, sample_2 = np.array_split(sample, 2)
                            sample_1.reset_index(inplace=True)
                            sample_2.reset_index(inplace=True)

                            # getting all other columns
                            column_list = [
                                column for column in sample if column.startswith("X")
                            ]
                            X1_sample_old = sample_1[column_list]
                            a1_sample = sample_1["A"]
                            y1_sample = sample_1["Y"]

                            X2_sample_old = sample_2[column_list]
                            a2_sample = sample_2["A"]
                            y2_sample = sample_2["Y"]

                            # sparse format
                            X1_sample = csr_matrix(X1_sample_old, dtype=float)
                            X2_sample = csr_matrix(X2_sample_old, dtype=float)

                            # fit on piece 1, estimate effect on piece 2
                            # if GAM or logistic GAM:
                            if (
                                model.get_treat_model().get_learner_string()
                                == "LogisticGAM"
                                or model.get_out_model().get_learner_string()
                                == "LogisticGAM"
                                or model.get_out_model().get_learner_string() == "GAM"
                                or model.get_treat_model().get_learner_string()
                                == "SuperLearner"
                                or model.get_out_model().get_learner_string()
                                == "SuperLearner"
                            ):
                                X1_sample = X1_sample_old
                                X2_sample = X2_sample_old

                            model.fit(X1_sample, a1_sample, y1_sample)
                            if len(self.truncate_percent) != 0:
                                (
                                    outcome_piece2_0,
                                    outcome_piece2_1,
                                ) = model.estimate_population_outcome(
                                    X2_sample,
                                    a2_sample,
                                    y2_sample,
                                    truncate=self.truncate_percent[0],
                                )
                            else:
                                (
                                    outcome_piece2_0,
                                    outcome_piece2_1,
                                ) = model.estimate_population_outcome(
                                    X2_sample, a2_sample, y2_sample
                                )
                            eff_piece2 = model.estimate_effect(
                                outcome_piece2_1, outcome_piece2_0, self.effect_types,
                            )

                            # fit on piece 2, estimate effect on piece 1
                            model.fit(X2_sample, a2_sample, y2_sample)
                            if len(self.truncate_percent) != 0:
                                (
                                    outcome_piece1_0,
                                    outcome_piece1_1,
                                ) = model.estimate_population_outcome(
                                    X1_sample,
                                    a1_sample,
                                    y1_sample,
                                    truncate=self.truncate_percent[0],
                                )
                            else:
                                (
                                    outcome_piece1_0,
                                    outcome_piece1_1,
                                ) = model.estimate_population_outcome(
                                    X1_sample, a1_sample, y1_sample
                                )
                            eff_piece1 = model.estimate_effect(
                                outcome_piece1_1, outcome_piece1_0, self.effect_types,
                            )

                            stats0 = stats0.append(
                                (outcome_piece2_0 + outcome_piece1_0) / 2
                            )
                            stats1 = stats1.append(
                                (outcome_piece2_1 + outcome_piece1_1) / 2
                            )

                            effs = effs.append((eff_piece1 + eff_piece2) / 2)

                        else:
                            dict1["sample split"] = "False"
                            model.fit(X_sample, a_sample, y_sample)
                            if len(self.truncate_percent) != 0:
                                (
                                    outcome0,
                                    outcome1,
                                ) = model.estimate_population_outcome(
                                    X_sample,
                                    a_sample,
                                    y_sample,
                                    truncate=self.truncate_percent[0],
                                )
                            else:
                                (
                                    outcome0,
                                    outcome1,
                                ) = model.estimate_population_outcome(
                                    X_sample, a_sample, y_sample
                                )

                            stats0 = stats0.append(outcome0, ignore_index=True)
                            stats1 = stats1.append(outcome1, ignore_index=True)
                            effs = effs.append(
                                model.estimate_effect(stats1, stats0, self.effect_types)
                            )

                if type(model) == DoublyRobust:
                    dict1 = {}
                    dict1["method"] = model.get_method()
                    dict1["std error"] = "bootstrap"
                    dict1["sample split"] = str(bool(self.sample_split))
                    dict1["outcome model"] = model.get_out_model_string()
                    dict1["treatment model"] = model.get_treat_model_string()
                    dict1["out learner"] = model.get_learner_string()[0]
                    dict1["treat learner"] = model.get_learner_string()[1]

                    if (
                        type(model.get_out_model().get_learner()).__name__
                        == "VotingRegressor"
                        or type(model.get_out_model().get_learner()).__name__
                        == "StackingRegressor"
                    ):
                        ens_learners = model.get_out_model().get_learner().estimators_
                        dict1["out ensemble learners"] = str(
                            [type(learn).__name__ for learn in ens_learners]
                        )

                    if (
                        type(model.get_treat_model().get_learner()).__name__
                        == "VotingRegressor"
                        or type(model.get_out_model().get_learner()).__name__
                        == "StackingRegressor"
                    ):
                        ens_learners = model.get_treat_model().get_learner().estimators_
                        dict1["treat ensemble learners"] = str(
                            [type(learn).__name__ for learn in ens_learners]
                        )

                    # treatment = 0
                    metrics = utils.compute_metrics(stats0)
                    dict1["a=0 mean"] = est0[0]
                    dict1["a=0 lower"] = metrics[CONF_95_LOWER]
                    dict1["a=0 upper"] = metrics[CONF_95_UPPER]

                    # treatment = 1
                    metrics = utils.compute_metrics(stats1)
                    dict1["a=1 mean"] = est1[0]
                    dict1["a=1 lower"] = metrics[CONF_95_LOWER]
                    dict1["a=1 upper"] = metrics[CONF_95_UPPER]

                    # effect estimate
                    eff_column = list(effs.columns.values)

                    # set avg_eff to mean for effect estimates
                    for col_name in eff_column:
                        metrics = utils.compute_metrics(effs[col_name])
                        stats_names = self.COLUMN_NAMES[col_name]
                        for key in stats_names:
                            if key == "mean":
                                dict1[stats_names[key]] = avg_eff[col_name][0]
                            else:
                                dict1[stats_names[key]] = metrics[key]

                    dict1["time"] = time.time() - start_time

                    # Get e-values for risk ratios and odds ratios
                    rr_evalues = get_evalue_rr(
                        dict1["ratio lower"], dict1["ratio mean"], dict1["ratio upper"],
                    )
                    dict1["ratio lower e-value"] = rr_evalues[0]
                    dict1["ratio mean e-value"] = rr_evalues[1]
                    dict1["ratio upper e-value"] = rr_evalues[2]

                    or_evalues = get_evalue_rr(
                        dict1["ratio lower"] ** 0.5,
                        dict1["ratio mean"] ** 0.5,
                        dict1["ratio upper"] ** 0.5,
                    )
                    dict1["odds ratio lower e-value"] = or_evalues[0]
                    dict1["odds ratio mean e-value"] = or_evalues[1]
                    dict1["odds ratio upper e-value"] = or_evalues[2]

                    # Get e-values for risk difference if binary, different in
                    # continuous outcomes if not
                    if self.outcome_type[0] == "binary":
                        # Risk difference
                        if (
                            dict1["a=1 mean"] > 0.2
                            and dict1["a=1 mean"] < 0.8
                            and dict1["a=0 mean"] > 0.2
                            and dict1["a=0 mean"] < 0.8
                        ):
                            # Same method as difference in continuous outcomes
                            d = dict1["diff mean"] / dict1["diff std"]
                            rd_evalues = get_evalue_rr(
                                math.exp(0.91 * d - 1.78 * dict1["diff std"]),
                                math.exp(0.91 * d),
                                math.exp(0.91 * d + 1.78 * dict1["diff std"]),
                            )
                            dict1["diff lower e-value"] = rd_evalues[0]
                            dict1["diff mean e-value"] = rd_evalues[1]
                            dict1["diff upper e-value"] = rd_evalues[2]
                        else:
                            rr = dict1["a=1 mean"] / dict1["a=0 mean"]
                            rd_evalues = get_evalue_rr(0, rr, rr * 2)
                            dict1["diff mean e-value"] = rd_evalues[1]
                    else:
                        # Difference in continuous outcomes
                        d = dict1["diff mean"] / dict1["diff std"]
                        dc_evalues = get_evalue_rr(
                            math.exp(0.91 * d - 1.78 * dict1["diff std"]),
                            math.exp(0.91 * d),
                            math.exp(0.91 * d + 1.78 * dict1["diff std"]),
                        )
                        dict1["diff lower e-value"] = dc_evalues[0]
                        dict1["diff mean e-value"] = dc_evalues[1]
                        dict1["diff upper e-value"] = dc_evalues[2]

                    row_list.append(dict1)

                else:
                    dict1 = {}
                    dict1["method"] = model.get_method()
                    dict1["std error"] = "bootstrap"
                    dict1["learner"] = str(model.get_learner_string())
                    if (
                        type(model.get_learner()).__name__ == "VotingRegressor"
                        or type(model.get_learner()).__name__ == "StackingRegressor"
                    ):
                        ens_learners = model.get_learner().estimators_
                        dict1["ensemble learners"] = str(
                            [type(learn).__name__ for learn in ens_learners]
                        )

                    # treatment = 0
                    metrics = utils.compute_metrics(stats0)
                    dict1["a=0 mean"] = metrics[MEAN]
                    dict1["a=0 lower"] = metrics[CONF_95_LOWER]
                    dict1["a=0 upper"] = metrics[CONF_95_UPPER]

                    # treatment = 1
                    metrics = utils.compute_metrics(stats1)
                    dict1["a=1 mean"] = metrics[MEAN]
                    dict1["a=1 lower"] = metrics[CONF_95_LOWER]
                    dict1["a=1 upper"] = metrics[CONF_95_UPPER]

                    # effect estimate
                    eff = model.estimate_effect(stats1, stats0, self.effect_types)
                    # print(eff)
                    # print(stats1)
                    # print(stats0)
                    eff_column = list(eff.columns.values)

                    # print(eff_column)
                    for col_name in eff_column:
                        metrics = utils.compute_metrics(eff[col_name])
                        stats_names = self.COLUMN_NAMES[col_name]
                        for key in stats_names:
                            dict1[stats_names[key]] = metrics[key]

                    dict1["time"] = time.time() - start_time

                    # Get e-values for risk ratios and odds ratios
                    rr_evalues = get_evalue_rr(
                        dict1["ratio lower"], dict1["ratio mean"], dict1["ratio upper"],
                    )
                    dict1["ratio lower e-value"] = rr_evalues[0]
                    dict1["ratio mean e-value"] = rr_evalues[1]
                    dict1["ratio upper e-value"] = rr_evalues[2]

                    or_evalues = get_evalue_rr(
                        dict1["ratio lower"] ** 0.5,
                        dict1["ratio mean"] ** 0.5,
                        dict1["ratio upper"] ** 0.5,
                    )
                    dict1["odds ratio lower e-value"] = or_evalues[0]
                    dict1["odds ratio mean e-value"] = or_evalues[1]
                    dict1["odds ratio upper e-value"] = or_evalues[2]

                    # Get e-values for risk difference if binary, different in
                    # continuous outcomes if not
                    if self.outcome_type[0] == "binary":
                        # Risk difference
                        if (
                            dict1["a=1 mean"] > 0.2
                            and dict1["a=1 mean"] < 0.8
                            and dict1["a=0 mean"] > 0.2
                            and dict1["a=0 mean"] < 0.8
                        ):
                            # Same method as difference in continuous outcomes
                            d = dict1["diff mean"] / dict1["diff std"]
                            rd_evalues = get_evalue_rr(
                                math.exp(0.91 * d - 1.78 * dict1["diff std"]),
                                math.exp(0.91 * d),
                                math.exp(0.91 * d + 1.78 * dict1["diff std"]),
                            )
                            dict1["diff lower e-value"] = rd_evalues[0]
                            dict1["diff mean e-value"] = rd_evalues[1]
                            dict1["diff upper e-value"] = rd_evalues[2]
                        else:
                            rr = dict1["a=1 mean"] / dict1["a=0 mean"]
                            rd_evalues = get_evalue_rr(0, rr, rr * 2)
                            dict1["diff mean e-value"] = rd_evalues[1]
                    else:
                        # Difference in continuous outcomes
                        d = dict1["diff mean"] / dict1["diff std"]
                        dc_evalues = get_evalue_rr(
                            math.exp(0.91 * d - 1.78 * dict1["diff std"]),
                            math.exp(0.91 * d),
                            math.exp(0.91 * d + 1.78 * dict1["diff std"]),
                        )
                        dict1["diff lower e-value"] = dc_evalues[0]
                        dict1["diff mean e-value"] = dc_evalues[1]
                        dict1["diff upper e-value"] = dc_evalues[2]

                    row_list.append(dict1)

                    row_list.append(dict1)

        return row_list


if __name__ == "__main__":
    input_dataset = "X_confounders.csv"
    config_file = "config_ipw_copy.json"
    import os

    dataset_no_extension = os.path.splitext(input_dataset)[0]
    master_model = MasterModel(config_file)

    dataset_pd = pd.read_csv(input_dataset)
    results = pd.DataFrame(master_model.process_dataset(dataset_pd))
    results.to_csv("master_model_results.csv")
