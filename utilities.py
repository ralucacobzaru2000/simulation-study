import json
import subprocess
import warnings
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Union

import numpy as np
import pandas as pd
import sklearn.ensemble as ensemble
import sklearn.linear_model as lm
import sklearn.neighbors as nghbs
import sklearn.neural_network as nn
import sklearn.svm as svm
import sklearn.tree as tree
from sklearn.utils.multiclass import unique_labels
from sklearn.base import BaseEstimator
from pygam import GAM, LogisticGAM
from sklearn.model_selection import GridSearchCV

from ipw3 import IPW3
from std3 import Standardization3

warnings.filterwarnings("ignore")
"""
import rpy2.robjects as robjects
from rpy2.robjects.packages import STAP
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
"""

import os
from sklearn.base import BaseEstimator
from scipy.sparse import csr_matrix

from scipy.optimize import fmin_l_bfgs_b, nnls, fmin_slsqp
from sklearn import clone
import sklearn.model_selection as ms


class ConstUtilities:
    def getInverseDict(self, dictionary: dict) -> dict:
        inverseDict = {dictionary[key]: key for key in dictionary}
        return inverseDict


class EmpiricalMean(BaseEstimator):
    """
    Calculates empirical mean
    """

    def __init__(self, est_type):
        self.empirical_mean = np.nan
        self._estimator_type = est_type

    def fit(self, X, y):
        self.empirical_mean = np.mean(y)

    def predict(self, X):
        return np.array([self.empirical_mean] * X.shape[0])


class Utilities:
    CONST_UTILS: ConstUtilities = ConstUtilities()

    N_NEIGHBORS: int = 50
    MIN_SAMPLES_LEAF: int = 100
    CONF_95: float = 1.96

    MEAN: str = "mean"
    STDEV: str = "standard deviation"
    CONF_INT_95: str = "95 % confidence interval"
    CONF_95_LOWER: str = "minimum of 95% confidence interval"
    CONF_95_UPPER: str = "maximum of 95% confidence interval"

    LEARNERS_TYPE: Dict[str, Any] = {
        "linear_regression": lm.LinearRegression,
        "lasso_CV": lm.LassoCV,
        "elastic_net_CV": lm.ElasticNetCV,
        "SVR_linear": svm.LinearSVR,
        "SVR_rbf": svm.SVR,
        "ada_boost_r": ensemble.AdaBoostRegressor,
        "decision_tree_r": tree.DecisionTreeRegressor,
        "gradient_boosting_r": ensemble.GradientBoostingRegressor,
        "KNeighbors_r": nghbs.KNeighborsRegressor,
        "MLP_r": nn.MLPRegressor,
        "random_forest_r": ensemble.RandomForestRegressor,
        "logistic_regression": lm.LogisticRegression,
        "ada_boost_c": ensemble.AdaBoostClassifier,
        "decision_tree_c": tree.DecisionTreeClassifier,
        "gradient_boosting_c": ensemble.GradientBoostingClassifier,
        "KNeighbors_c": nghbs.KNeighborsClassifier,
        "MLP_c": nn.MLPClassifier,
        "random_forest_c": ensemble.RandomForestClassifier,
        "SVC": svm.SVC,
        "voting_c": ensemble.VotingClassifier,
        "stacking_c": ensemble.StackingClassifier,
        "voting_r": ensemble.VotingRegressor,
        "stacking_r": ensemble.StackingRegressor,
        "logistic_GAM": LogisticGAM,
        "GAM": GAM,
        "empirical_mean_r": EmpiricalMean,
        "empirical_mean_c": EmpiricalMean,
    }

    LEARNERS_TYPE_NAME: Dict[Any, str] = CONST_UTILS.getInverseDict(LEARNERS_TYPE)

    LEARNER_PARAMS = {
        "linear_regression": {"fit_intercept": True, "normalize": False},
        "logistic_regression": {"fit_intercept": True, "penalty": "none"},
        "KNeighbors_c": {"n_neighbors": N_NEIGHBORS},
        "SVC": {"probability": True},
        "voting_c": {"estimators": None, "voting": "soft"},
        "stacking_c": {"estimators": None},
        "voting_r": {"estimators": None},
        "stacking_r": {"estimators": None},
        "decision_tree_r": {"min_samples_leaf": MIN_SAMPLES_LEAF},
        "decision_tree_c": {"min_samples_leaf": MIN_SAMPLES_LEAF},
        "empirical_mean_r": {"est_type": "regressor"},
        "empirical_mean_c": {"est_type": "classifier"},
    }
    # initialize default params to empty dict for unspecified learners
    LEARNER_PARAMS: DefaultDict[str, Dict[str, Any]] = defaultdict(
        lambda: {}, LEARNER_PARAMS
    )

    LEARNER_PARAM_GRID = {
        lm.LinearRegression: {
            "fit_intercept": [True, False],
            "normalize": [True, False],
            "copy_X": [True, False],
        },
        svm.SVC: {"kernel": ("linear", "rbf"), "C": [1, 10]},
        lm.LassoCV: {
            "eps": [],
            "n_alphas": [1, 0.1, 0.01, 0.001, 0.0001, 0],
            "fit_intercept": [True, False],
            "normalize": [True, False],
            "precompute": [],
            "max_iter": [],
            "tol": [],
            "copy_X": [True, False],
            "cv": [],
            "positive": [True, False],
            "random_state": [],
            "selection": ["cyclic", "random"],
        },
    }
    LEARNER_PARAM_GRID: DefaultDict[Any, Dict[str, Any]] = defaultdict(
        lambda: {}, LEARNER_PARAM_GRID
    )

    # user determined model, read in config
    def get_json_data(self, file_name: str) -> Dict[str, Any]:
        """
        User determine methods and learners, read in config
        """

        with open(file_name, "r") as document:
            json_obj = json.load(document)
            return json_obj

    def find_best_default_params(self, model, source_data, target):

        # Performs GridSearchCV to find best default params for each learner
        # Params: model: scikit learn object
        # source_data: X concatenated with A
        # target: Y column
        # Return: instance of scikit learner with best params

        model_type_name = self.LEARNERS_TYPE_NAME[type(model)]
        model_params = self.LEARNER_PARAMS[model_type_name]
        model = self.LEARNERS_TYPE[model_type_name](**model_params)
        param_grid = self.LEARNER_PARAM_GRID[type(model)]

        grid_result = GridSearchCV(estimator=model, param_grid=param_grid)
        grid_result.fit(source_data, target)
        best_estimator = grid_result.best_estimator_
        return best_estimator

    def get_learner(self, input_learner: str, estimators=None, y=None):
        learner = self.LEARNERS_TYPE[input_learner]
        learner_params = self.LEARNER_PARAMS[input_learner]

        estimators_name = "estimators"
        if estimators_name in learner_params:
            learner_params[estimators_name] = estimators
        if learner == LogisticGAM:
            learner.classes_ = [0, 1]
            learner = learner(**learner_params)
            learner._estimator_type = "classifier"
        elif learner == GAM:
            learner = learner(**learner_params)
            learner._estimator_type = "regressor"
        else:
            learner = learner(**learner_params)

        return learner

    def get_model(self, model_name: str, model_learner):
        # model_name is a string describing the model
        # model_learner is the learner instance
        model_dict = {
            "IPW": IPW3,
            "standardization": Standardization3,
        }

        return model_dict[model_name](learner=model_learner)

    def compute_metrics(self, results: pd.DataFrame):
        # results is pandas dataframe
        arr = np.array(results)
        mean = np.mean(arr)
        stdev = np.std(arr)

        conf_interval_95 = (
            mean - (self.CONF_95 * stdev),
            mean + (self.CONF_95 * stdev),
        )

        metrics = {
            self.MEAN: mean,
            self.STDEV: stdev,
            self.CONF_INT_95: conf_interval_95,
            self.CONF_95_LOWER: conf_interval_95[0],
            self.CONF_95_UPPER: conf_interval_95[1],
        }

        return metrics

    def build_model_list(self, learners: List[List[str]], model_str: str, y=None):
        """
        Build list of models to use in the adjustment methods
        Params: learners is list of list of learners
                if nested list length >= 1, is ensemble
        """
        models = []
        for learner in learners:
            # learner could be list of one or more items (if ensemble)
            if len(learner) > 1:
                if learner[0] in {"stacking_c", "stacking_r"}:
                    estimators = []
                    for i in range(1, len(learner)):
                        estimator_name = learner[i][0]
                        unique_estimator_name = f"{estimator_name}_{i}"

                        estimator = self.get_learner(estimator_name)
                        if len(learner[i]) > 1:
                            if learner[0].startswith("MLP"):
                                learner[1]["hidden_layer_sizes"] = tuple(
                                    learner[1]["hidden_layer_sizes"]
                                )
                            if learner[0] == "logistic_GAM":
                                estimator.n_splines = learner[1]["n_splines"]
                                estimator.lam = learner[1]["lam"]
                            if learner[0] == "GAM":
                                estimator.n_splines = learner[1]["n_splines"]
                                estimator.lam = learner[1]["lam"]
                            estimator.set_params(**learner[i][1])
                        estimators.append((unique_estimator_name, estimator))

                    model = self.get_model(
                        model_str, self.get_learner(learner[0], estimators)
                    )
                    models.append(model)

                elif learner[0] == "superlearner_c" or learner[0] == "superlearner_r":
                    lib = []
                    libnames = []
                    for i in range(1, len(learner)):
                        estimator_name = learner[i][0]
                        unique_estimator_name = f"{estimator_name}_{i}"
                        libnames.append(unique_estimator_name)
                        estimator = self.get_learner(estimator_name)
                        if len(learner[i]) > 1:
                            if learner[0].startswith("MLP"):
                                learner[1]["hidden_layer_sizes"] = tuple(
                                    learner[1]["hidden_layer_sizes"]
                                )
                            if learner[0] == "logistic_GAM":
                                estimator.n_splines = learner[1]["n_splines"]
                                estimator.lam = learner[1]["lam"]
                            if learner[0] == "GAM":
                                estimator.n_splines = learner[1]["n_splines"]
                                estimator.lam = learner[1]["lam"]
                            estimator.set_params(**learner[i][1])
                        lib.append(estimator)

                    var_type = ""
                    if learner[0] == "superlearner_c":
                        var_type = "binary"
                    elif learner[0] == "superlearner_r":
                        var_type = "continuous"
                    super_learner = super_learner_init(lib, libnames, var_type, K=10)
                    model = self.get_model(model_str, super_learner)
                    models.append(model)
                else:
                    # has specified params
                    estimator = self.get_learner(learner[0])
                    if len(learner) == 2:
                        if learner[0].startswith("MLP"):
                            learner[1]["hidden_layer_sizes"] = tuple(
                                learner[1]["hidden_layer_sizes"]
                            )
                        if learner[0] == "logistic_GAM":
                            estimator.n_splines = learner[1]["n_splines"]
                            estimator.lam = learner[1]["lam"]
                        if learner[0] == "GAM":
                            estimator.n_splines = learner[1]["n_splines"]
                            estimator.lam = learner[1]["lam"]

                        estimator.set_params(**learner[1])
                    model = self.get_model(model_str, estimator)

                    models.append(model)

            elif len(learner) == 1:
                model = self.get_model(model_str, self.get_learner(learner[0]))
                models.append(model)

        return models, estimator

    def us_split(self, y, letter, data_type="str"):
        # Returns a dataframe where the X column is split by the letter before the underscore
        # print('Making ' + letter)
        return y[y.x.str.split(pat="_", expand=True)[0] == letter].astype(
            {"event_value": data_type}
        )

    def merge(self, dflist):
        # Merges a list of dataframes into one via an outer join on the patient_id column
        # print('Merging')
        output = dflist[0]
        for i in range(len(dflist) - 1):
            output = pd.merge(output, dflist[i + 1], on="patient_id", how="outer")
        return output

    def rename(self, df, name):
        # Renames the X column
        return df[["patient_id", "event_value"]].rename(columns={"event_value": name})

    def dummify(self, df):
        # Changes a table into a form with dummy columns for each X value and indicators
        return (
            pd.get_dummies(data=df, columns=["x"], prefix=["X"], dtype=bool)
            .groupby("patient_id")
            .sum()
            .astype("bool")
            .astype("int8")
        )

    def decompress(self, file_in, file_out=None):
        # Turns a compressed cohort into a decompressed cohort
        y = pd.read_csv(file_in)
        output = self.merge(
            [
                self.rename(self.us_split(y, "r", "int8"), "A"),
                self.rename(self.us_split(y, "s", "int8"), "Y_indicator"),
                self.rename(self.us_split(y, "f", "int16"), "Y_time"),
                self.dummify(self.us_split(y, "i")),
                self.dummify(self.us_split(y, "d")),
                self.dummify(self.us_split(y, "p")),
            ]
        ).fillna(0)
        if file_out == None:
            return output.to_csv(index=False, line_terminator="\n")
        goal_dir = os.path.join(os.getcwd(), file_out)
        return output.to_csv(
            os.path.abspath(goal_dir), index=False, line_terminator="\n"
        )

    def covar_keep_sparse(self, file_in):
        # keeps sparse format matrix and returns time, event, A, covar in sparse format
        # event, treat not in sparse format
        y = pd.read_csv(file_in)
        time = self.rename(self.us_split(y, "f", "int16"), "Y_time")["Y_time"].fillna(0)
        event = self.rename(self.us_split(y, "s", "int8"), "Y_indicator")[
            "Y_indicator"
        ].fillna(0)
        A = self.rename(self.us_split(y, "r", "int8"), "A")["A"].fillna(0)

        covar_ones_df = pd.concat(
            [self.us_split(y, "p"), self.us_split(y, "i"), self.us_split(y, "d")],
            axis=0,
        )
        num_ones = len(covar_ones_df)

        covar_names = pd.unique(covar_ones_df["x"].values.ravel())

        col_index = pd.Series(np.arange(len(covar_names)), covar_names)

        patient_ids = pd.unique(y["patient_id"].values.ravel())

        row_index = pd.Series(np.arange(len(patient_ids)), patient_ids)

        covar_ones_df[["row_index"]] = covar_ones_df.applymap(row_index.get)[
            "patient_id"
        ]
        covar_ones_df[["col_index"]] = covar_ones_df.applymap(col_index.get)["x"]

    def covar_keep_sparse(self, file_in):
        # keeps sparse format matrix and returns time, event, A, covar in sparse format
        # event, treat not in sparse format
        y = pd.read_csv(file_in)
        time = self.rename(self.us_split(y, "f", "int16"), "Y_time")["Y_time"].fillna(0)
        event = self.rename(self.us_split(y, "s", "int8"), "Y_indicator")[
            "Y_indicator"
        ].fillna(0)
        A = self.rename(self.us_split(y, "r", "int8"), "A")["A"].fillna(0)

        covar_ones_df = pd.concat(
            [self.us_split(y, "p"), self.us_split(y, "i"), self.us_split(y, "d")],
            axis=0,
        )
        num_ones = len(covar_ones_df)

        covar_names = pd.unique(covar_ones_df["x"].values.ravel())

        col_index = pd.Series(np.arange(len(covar_names)), covar_names)

        patient_ids = pd.unique(y["patient_id"].values.ravel())

        row_index = pd.Series(np.arange(len(patient_ids)), patient_ids)

        covar_ones_df[["row_index"]] = covar_ones_df.applymap(row_index.get)[
            "patient_id"
        ]
        covar_ones_df[["col_index"]] = covar_ones_df.applymap(col_index.get)["x"]

        data = [1] * num_ones
        rows = covar_ones_df["col_index"].tolist()
        cols = covar_ones_df["row_index"].tolist()

        covar_sparse = csr_matrix((data, (cols, rows)), shape=(len(rows), len(cols)))
        return time, event, A, covar_sparse

    def decompress_to_output_file(
        self, config_file, check_decompress_binary, file_in_name
    ):
        # config file: string of config file
        # check_decompress_binary: value of 0 or 1:
        # if 1, then decompress
        # if 0, then keep compressed
        # file_in_name: string of input file name
        # returns: file_out_name: string of output file name with decompressed matrix

        input_dict = self.get_json_data(config_file)
        check_decompress = input_dict["advanced"]["data_process"]["decompress"]

        if check_decompress_binary == 1:  # then decompress
            # file_in_name_base = file_in_name.split('.')[0]
            file_in_name_base = os.path.splitext(file_in_name)[0]
            file_in_name_ext = os.path.splitext(file_in_name)[1]
            # print(file_in_name_base)
            # print(file_in_name_ext)

            file_out_name = "{}_{}{}".format("de", file_in_name_base, file_in_name_ext)
            # print(file_out_name)
            self.decompress(file_in=file_in_name, file_out=file_out_name)
            return file_out_name


class SLError(Exception):
    """
    Base class for errors in the SupyLearner package
    """

    pass


class SuperLearner(BaseEstimator):
    """Loss-based super learning. SuperLearner chooses a weighted combination of candidate estimates in a specified
    library using cross-validation.
    Implementation borrowed from: https://github.com/alexpkeil1/SuPyLearner
    Parameters
    ----------
    library : list
        List of scikit-learn style estimators with fit() and predict()
        methods.
    K : Number of folds for cross-validation.
    loss : loss function, 'L2' or 'nloglik'.
    discrete : True to choose the best estimator
               from library ("discrete SuperLearner"), False to choose best
               weighted combination of esitmators in the library.
    coef_method : Method for estimating weights for weighted combination
                  of estimators in the library. 'L_BFGS_B', 'NNLS', or 'SLSQP'.
    """

    def __init__(
        self,
        library,
        libnames=None,
        K=10,
        loss="L2",
        discrete=False,
        coef_method="SLSQP",
        save_pred_cv=False,
        bound=0.00001,
        print_results=True,
    ):
        self.library = library[:]
        self.libnames = libnames
        self.K = K
        self.loss = loss
        self.n_estimators = len(library)
        self.discrete = discrete
        self.coef_method = coef_method
        self.save_pred_cv = save_pred_cv
        self.bound = bound
        self._print = print_results

    def get_learner_string(self):
        return type(self).__name__

    def fit(self, X, y):
        """Fit SuperLearner.
        Parameters
        ----------
        X : numpy array of shape [n_samples,n_features]
            or other object acceptable to the fit() methods
            of all candidates in the library
            Training data
        y : numpy array of shape [n_samples]
            Target values
        Returns
        -------
        self : returns an instance of self.
        """
        n = len(y)
        folds = ms.KFold(self.K)

        y_pred_cv = np.empty(shape=(n, self.n_estimators))
        for train_index, test_index in folds.split(range(n)):
            if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            else:
                print(type(X))
                print(X[train_index])
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
            for aa in range(self.n_estimators):
                if self.libnames is not None:
                    if self._print:
                        print("...K-fold fitting " + self.libnames[aa] + "...")
                est = clone(self.library[aa])
                est.fit(X_train, y_train)

                y_pred_cv[test_index, aa] = self._get_pred(est, X_test)

        self.coef = self._get_coefs(y, y_pred_cv)

        self.fitted_library = clone(self.library)

        for est in self.fitted_library:
            ind = self.fitted_library.index(est)
            if self._print:
                print("...fitting " + self.libnames[ind] + "...")
            est.fit(X, y)

        self.risk_cv = []
        for aa in range(self.n_estimators):
            self.risk_cv.append(self._get_risk(y, y_pred_cv[:, aa]))
        self.risk_cv.append(
            self._get_risk(y, self._get_combination(y_pred_cv, self.coef))
        )

        if self.save_pred_cv:
            self.y_pred_cv = y_pred_cv

        return self

    def predict(self, X):
        """Predict using SuperLearner
        Parameters
        ----------
        X : numpy.array of shape [n_samples, n_features]
           or other object acceptable to the predict() methods
           of all candidates in the library
        Returns
        -------
        array, shape = [n_samples]
           Array containing the predicted class labels.
        """

        n_X = X.shape[0]
        y_pred_all = np.empty((n_X, self.n_estimators))
        for aa in range(self.n_estimators):
            y_pred_all[:, aa] = self._get_pred(self.fitted_library[aa], X)
        y_pred = self._get_combination(y_pred_all, self.coef)
        return y_pred

    def summarize(self):
        """Print CV risk estimates for each candidate estimator in the library,
        coefficients for weighted combination of estimators,
        and estimated risk for the SuperLearner.
        """
        if self.libnames is None:
            libnames = [est.__class__.__name__ for est in self.library]
        else:
            libnames = self.libnames
        print("Cross-validated risk estimates for each estimator in the library:")
        print(np.column_stack((libnames, self.risk_cv[:-1])))
        print("\nCoefficients:")
        print(np.column_stack((libnames, self.coef)))
        print("\n(Not cross-valided) estimated risk for SL:", self.risk_cv[-1])

    def _get_combination(self, y_pred_mat, coef):
        """Calculate weighted combination of predictions
        """
        if self.loss == "L2":
            comb = np.dot(y_pred_mat, coef)
        elif self.loss == "nloglik":
            comb = _inv_logit(np.dot(_logit(_trim(y_pred_mat, self.bound)), coef))
        return comb

    def _get_risk(self, y, y_pred):
        """Calculate risk given observed y and predictions
        """
        if self.loss == "L2":
            risk = np.mean((y - y_pred) ** 2)
        elif self.loss == "nloglik":
            risk = -np.mean(
                y * np.log(_trim(y_pred, self.bound))
                + (1 - y) * np.log(1 - (_trim(y_pred, self.bound)))
            )
        return risk

    def _get_coefs(self, y, y_pred_cv):
        """Find coefficients that minimize the estimated risk.
        """
        if self.coef_method is "L_BFGS_B":
            if self.loss == "nloglik":
                raise SLError("coef_method 'L_BFGS_B' is only for 'L2' loss")

            def ff(x):
                return self._get_risk(y, self._get_combination(y_pred_cv, x))

            x0 = np.array([1.0 / self.n_estimators] * self.n_estimators)
            bds = [(0, 1)] * self.n_estimators
            coef_init, b, c = fmin_l_bfgs_b(ff, x0, bounds=bds, approx_grad=True)
            if c["warnflag"] is not 0:
                raise SLError(
                    "fmin_l_bfgs_b failed when trying to calculate coefficients"
                )

        elif self.coef_method is "NNLS":
            if self.loss == "nloglik":
                raise SLError("coef_method 'NNLS' is only for 'L2' loss")
            coef_init, b = nnls(y_pred_cv, y)

        elif self.coef_method is "SLSQP":

            def ff(x):
                return self._get_risk(y, self._get_combination(y_pred_cv, x))

            def constr(x):
                return np.array([np.sum(x) - 1])

            x0 = np.array([1.0 / self.n_estimators] * self.n_estimators)
            bds = [(0, 1)] * self.n_estimators
            coef_init, b, c, d, e = fmin_slsqp(
                ff, x0, f_eqcons=constr, bounds=bds, disp=0, full_output=True
            )
            if d is not 0:
                raise SLError("fmin_slsqp failed when trying to calculate coefficients")

        else:
            raise ValueError("method not recognized")
        coef_init = np.array(coef_init)
        coef_init[coef_init < np.sqrt(np.finfo(np.double).eps)] = 0
        coef = coef_init / np.sum(coef_init)
        return coef

    def _get_pred(self, est, X):
        """
        Get prediction from the estimator.
        Use est.predict if loss is L2.
        If loss is nloglik, use est.predict_proba if possible
        otherwise just est.predict, which hopefully returns something
        like a predicted probability, and not a class prediction.
        """
        if self.loss == "L2":
            pred = est.predict(X)
        elif self.loss == "nloglik":
            if hasattr(est, "predict_proba"):
                try:
                    pred = est.predict_proba(X)[:, 1]
                except IndexError:
                    pred = est.predict_proba(X)
            else:
                pred = est.predict(X)
                if pred.min() < 0 or pred.max() > 1:
                    raise SLError("Probability less than zero or greater than one")
        else:
            raise SLError("loss must be 'L2' or 'nloglik'")
        return pred


def _trim(p, bound):
    """Trim a probabilty to be in (bound, 1-bound)
    """
    p[p < bound] = bound
    p[p > 1 - bound] = 1 - bound
    return p


def _logit(p):
    """Calculate the logit of a probability
    """
    return np.log(p / (1 - p))


def _inv_logit(x):
    """Calculate the inverse logit
    """

    return 1 / (1 + np.exp(-x))


def super_learner_init(lib, libnames, var_type, K=10):
    # super_learner_list: list of lists [["logreg", {}], ["mlp_r", {}]]
    """Super Learner setup for binary and continuous variables"""
    if var_type == "binary":
        sl = SuperLearner(lib, libnames, loss="nloglik", K=K, print_results=False)
    if var_type == "continuous":
        sl = SuperLearner(lib, libnames, K=K, print_results=False)
    return sl
