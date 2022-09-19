import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import bernoulli, norm
from ascvd_dgp import PatientDGP


def gen_noise_vars(
    n_noise: int,
    n_size: int,
    n_original: int,
    parameter: float,
    noise_type="continuous",
) -> pd.DataFrame:
    """
    Generates dataframe of noise variables which can be added to main
    dataframe or other bias-generating functions
    ---
    n_noise : int
              number of noise variables to generate
    n_size: : int
              number of observations in dataset
    n_original : int
                 number of X columns in original dataset
    noise_type : string
                 type of noise variables
                 ("continuous", "bernoulli")
    parameter : float
                parameters of generated noise variables
                (mean for Boolean, SD for continuous)
    seed : int
           seed for generating variables
    """
    if noise_type == "bernoulli":
        confounding_cols = [
            bernoulli.rvs(expit(parameter), size=n_size) for i in range(n_noise)
        ]
    else:
        confounding_cols = [
            norm.rvs(loc=0, scale=parameter, size=n_size) for i in range(n_noise)
        ]
    col_names = [f"X{i+1}" for i in range(n_original, n_original + n_noise)]
    confounding_df = pd.DataFrame(np.array(confounding_cols), col_names)
    return confounding_df.T


def add_noise(
    data: pd.DataFrame,
    n_vars: int,
    prop_bool: float,
    extra_params: float,
    large_noise: bool,
) -> pd.DataFrame:
    """
    Adds noise variables to dataset
    ---
    data : pd.DataFrame
           input dataset (columns X{int}, A, Y)
    n_vars : int
             number of noise variables to add
    large_noise : bool
                  if added noise, indicates if large noise
    prop_bool : float
                generate prop_bool Boolean, 1-prop_bool continuous variables
    extra_params : float
                   parameters of generated noise variables
                   (mean for Boolean, SD for continuous)
    """
    X_list = [column for column in data if column.startswith("X")]
    n_size = len(data)

    # number of variables in original dataset
    n_original = len(X_list)

    # number of variables of each type
    n_bool = int(prop_bool * n_vars)
    n_continuous = n_vars - n_bool

    if large_noise:
        ber_loc = extra_params[1]
        cont_sd = extra_params[3]
    else:
        ber_loc = extra_params[0]
        cont_sd = extra_params[2]

    df_continuous = gen_noise_vars(
        n_noise=n_continuous,
        n_size=n_size,
        n_original=n_original,
        noise_type="continuous",
        parameter=cont_sd,
    )

    df_bernoulli = gen_noise_vars(
        n_noise=n_bool,
        n_size=n_size,
        n_original=n_original + n_continuous,
        noise_type="bernoulli",
        parameter=ber_loc,
    )

    data = pd.concat([data, df_continuous, df_bernoulli], axis=1, sort=True)

    cols = data.columns.tolist()
    new_cols = (
        cols[0:n_original] + cols[n_original + 2 :] + cols[n_original : n_original + 2]
    )
    data = data.reindex(columns=new_cols)

    return data


def gen_exposure_vars(
    data: pd.DataFrame,
    n_vars: int,
    prop_bool: float,
    extra_params: float,
    strong_assoc: bool,
) -> pd.DataFrame:
    """
    Generates dataframe of variables correlated with the exposure variable
    which can be added to main dataframe or other bias-generating functions
    ---
    data : pd.DataFrame
           input dataset (columns X{int}, A, Y)
    n_vars : int
             number of correlated exposure variables to generate
    prop_bool : float
                generate prop_bool Boolean, 1-prop_bool continuous variables
    extra_params : float
                   parameters of generated noise variables
                   (mean for Boolean, SD for continuous)
    strong_assoc : bool
                   indicates if association with exposure is strong
    """
    X_list = [column for column in data if column.startswith("X")]
    n_size = len(data)
    exposure_var = data["A"]

    # number of variables in original dataset
    n_original = len(X_list)

    # number of variables of each type
    n_bool = int(prop_bool * n_vars)

    # beta in logistic dgp for binary variables and sd for continuous variables
    if strong_assoc:
        bool_beta = extra_params[1]
        cont_sd = extra_params[3]
    else:
        bool_beta = extra_params[0]
        cont_sd = extra_params[2]

    confounding_cols = np.zeros((n_size, n_vars))
    for i in range(n_vars):
        confounding_cols[:, i] = bernoulli.rvs(
            expit(bool_beta * exposure_var), size=n_size
        )

    # generating intermediate variable U
    U = bernoulli.rvs(expit(bool_beta * exposure_var), size=n_size)

    # generating the boolean variables X using U
    for i in range(n_bool):
        confounding_cols[:, i] = bernoulli.rvs(expit(bool_beta * U), size=n_size)

    for i in range(n_bool, n_vars):
        confounding_cols[:, i] = U

    col_names = [f"X{i+1}" for i in range(n_original, n_original + n_vars)]
    df_corr = pd.DataFrame(np.array(confounding_cols).T, col_names).T

    df_bool = df_corr[df_corr.columns[:n_bool]]
    df_cont = df_corr[df_corr.columns[n_bool:]]

    # generating the continuous variables X using U
    df_noise = gen_noise_vars(n_vars - n_bool, n_size, n_original + n_bool, cont_sd)
    df_cont = df_cont + df_noise
    df_corr = pd.concat([df_bool, df_cont], axis=1)

    return df_corr


def gen_outcome_vars(
    data: pd.DataFrame,
    cf_var: pd.Series,
    n_vars: int,
    prop_bool: float,
    extra_params: float,
    strong_assoc: bool,
) -> pd.DataFrame:
    """
    Generates dataframe of variables correlated with the outcome variable
    which can be added to main dataframe or other bias-generating functions
    ---
    data : pd.DataFrame
           input dataset (columns X{int}, A, Y)
    cf_var : pd.Series
             counterfactual outcome if a=0
    n_vars : int
             number of correlated outcome variables to generate
    prop_bool : float
                generate prop_bool Boolean, 1-prop_bool continuous variables
    extra_params : float
                   parameters of generated outcome and noise variables
                   (mean for Boolean, SD for continuous)
    strong_assoc : bool
                   indicates if association with outcome is strong
    """
    X_list = [column for column in data if column.startswith("X")]
    n_size = len(data)
    outcome_var = cf_var

    # number of variables in original dataset
    n_original = len(X_list)

    # number of variables of each type
    n_bool = int(prop_bool * n_vars)

    # beta in logistic dgp for binary variables and sd for continuous variables
    if strong_assoc:
        bool_beta = extra_params[1]
        cont_sd = extra_params[3]
    else:
        bool_beta = extra_params[0]
        cont_sd = extra_params[2]

    confounding_cols = np.zeros((n_size, n_vars))
    # generating intermediate variable U
    U = bernoulli.rvs(expit(bool_beta * outcome_var), size=n_size)
    # generating the boolean variables X using U
    for i in range(n_bool):
        confounding_cols[:, i] = bernoulli.rvs(expit(bool_beta * U), size=n_size)

    for i in range(n_bool, n_vars):
        confounding_cols[:, i] = U

    col_names = [f"X{i+1}" for i in range(n_original, n_original + n_vars)]
    df_corr = pd.DataFrame(np.array(confounding_cols).T, col_names).T

    df_bool = df_corr[df_corr.columns[:n_bool]]
    df_cont = df_corr[df_corr.columns[n_bool:]]

    # generating the continuous variables X
    df_noise = gen_noise_vars(n_vars - n_bool, n_size, n_original + n_bool, cont_sd)
    df_cont = df_cont + df_noise
    df_corr = pd.concat([df_bool, df_cont], axis=1)

    return df_corr


def gen_M_vars(
    data: pd.DataFrame,
    cf_var: pd.Series,
    n_vars: int,
    prop_bool: float,
    extra_params: float,
    strong_assoc: bool,
) -> pd.DataFrame:
    """
    Adds collider variables within an M-bias DAG to dataset
    ---
    data : pd.DataFrame
           input dataset (columns X{int}, A, Y)
    cf_var : pd.Series
             counterfactual outcome if a=0
    n_vars : int
             number of correlated M-bias variables to generate
    prop_bool : float
                generate prop_bool Boolean, 1-prop_bool continuous variables
    extra_params : float
                   parameters of generated M and noise variables
                   (mean for Boolean, SD for continuous)
    strong_assoc : bool
                   indicates if association with exposure and outcome is strong
    """
    X_list = [column for column in data if column.startswith("X")]
    n_size = len(data)

    # number of variables in original dataset
    n_original = len(X_list)

    # number of variables of each type
    n_bool = int(prop_bool * n_vars)

    # beta in logistic dgp for binary variables and sd for continuous variables
    # for the M-bias colliders
    if strong_assoc:
        bool_beta = extra_params[9]
        cont_sd = extra_params[11]
    else:
        bool_beta = extra_params[8]
        cont_sd = extra_params[10]

    exposure_var = data["A"]
    outcome_var = cf_var

    # generating hidden U1, U2
    U1 = bernoulli.rvs(expit(bool_beta * exposure_var), size=n_size)
    U2 = bernoulli.rvs(expit(bool_beta * outcome_var), size=n_size)

    confounding_cols = np.zeros((n_size, n_vars))
    # generating the boolean variables X
    for i in range(n_bool):
        confounding_cols[:, i] += bernoulli.rvs(
            expit(bool_beta * (U1 + U2)), size=n_size,
        )

    # generating continuous variables
    for i in range(n_bool, n_vars):
        confounding_cols[:, i] = U1 + U2

    col_names = [f"X{i+1}" for i in range(n_original, n_original + n_vars)]
    df_corr = pd.DataFrame(np.array(confounding_cols).T, col_names).T

    df_bool = df_corr[df_corr.columns[:n_bool]]
    df_cont = df_corr[df_corr.columns[n_bool:]]

    df_noise = gen_noise_vars(n_vars - n_bool, n_size, n_original + n_bool, cont_sd)
    df_cont = df_cont + df_noise
    df_corr = pd.concat([df_bool, df_cont], axis=1)

    return df_corr


def add_variables(
    data: pd.DataFrame,
    extra_params: float,
    type: str,
    cf_var=None,
    n_vars=0,
    large_noise=False,
    strong_assoc=False,
    prop_bool=0.5,
) -> pd.DataFrame:
    """
    Adds differently correlated variables to given dataset
    ---
    data : pd.DataFrame
           input dataset (columns X{int}, A, Y)
    cf_var : pd.Series
             counterfactual outcome if a=0
    n_vars : int
              number of generated variables to add
    type : str
           type of added variables (noise, exposure, outcome, M-bias)
    large_noise : bool
                  if added noise, indicates if large noise
    strong_assoc : bool
                   if added exposure, outcome, or M-bias type, indicates
                   if association is strong
    prop_bool : float
                generate prop_bool Boolean, 1-prop_bool continuous variables
    extra_params : float
                   parameters used in the DGP
    """

    X_list = [column for column in data if column.startswith("X")]
    # number of variables in original dataset
    n_original = len(X_list)

    if type == "noise":
        data = add_noise(
            data=data,
            n_vars=n_vars,
            prop_bool=prop_bool,
            extra_params=extra_params,
            large_noise=large_noise,
        )
    if type == "exposure":
        exposure_df = gen_exposure_vars(
            data=data,
            n_vars=n_vars,
            prop_bool=prop_bool,
            extra_params=extra_params,
            strong_assoc=strong_assoc,
        )
        data = pd.concat([data, exposure_df], axis=1, sort=True)
    if type == "outcome":
        outcome_df = gen_outcome_vars(
            data=data,
            n_vars=n_vars,
            cf_var=cf_var,
            prop_bool=prop_bool,
            extra_params=extra_params,
            strong_assoc=strong_assoc,
        )
        data = pd.concat([data, outcome_df], axis=1, sort=True)
    if type == "M-bias":
        M_df = gen_M_vars(
            data=data,
            n_vars=n_vars,
            cf_var=cf_var,
            prop_bool=prop_bool,
            extra_params=extra_params,
            strong_assoc=strong_assoc,
        )
        data = pd.concat([data, M_df], axis=1, sort=True)

    if type != "noise":
        cols = data.columns.tolist()
        new_cols = (
            cols[0:n_original]
            + cols[n_original + 2 :]
            + cols[n_original : n_original + 2]
        )
        data = data.reindex(columns=new_cols)

    return data


if __name__ == "__main__":
    sample_size = 3000  # size of each sample
    sim_dict = PatientDGP().simulate_patient(sample_size, True)
    orig_dataset = sim_dict["data"]
    orig_dataset.to_csv("confounder_test_original.csv", index=False)

    confounder_dataset = add_variables(
        orig_dataset,
        n_vars=1000,
        type="outcome",
        strong_assoc=True,
        prop_bool=0.2,
        extra_params=[0.7, 1.8, 2, 0.5, 0.15, 0.5, 2, 0.5, 0.7, 1.65, 1.75, 0.8],
        cf_var=sim_dict["Y_0"],
    )

    confounder_dataset.to_csv("confounders_final.csv")
