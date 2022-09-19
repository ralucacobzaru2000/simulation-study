import numpy as np
from scipy.special import expit
from scipy.stats import bernoulli, norm, uniform
import pandas as pd


class PatientDGP:
    def sim_age(self, v: float) -> int:
        return (75 - np.sqrt(30 * (v - 60) * (v > 60))) * (v > 60) + v * (v <= 60)

    def sim_lipo(self, n: int, a: int) -> float:
        # return 0.005 * a + norm.rvs(loc=np.log(100), scale=np.sqrt(0.18), size=n)
        return 0.005 * a + norm.rvs(loc=np.log(100), scale=0.18, size=n)

    def sim_diabetes(self, n: int, a: int, l: float) -> float:
        pbs = expit(-4.23 + 0.03 * l - 0.02 * a + 0.0009 * a ** 2)
        return np.array([bernoulli.rvs(pbs[i]) for i in range(n)])

    def sim_frailty(self, n: int, a: int) -> float:
        return expit(
            -5.5 + 0.05 * (a - 20) + 0.001 * a ** 2 + norm.rvs(loc=0, scale=1, size=n)
        )

    def sim_risk(self, a: int, l: float, d: float, f: float) -> float:
        lage = np.log(a)
        return expit(
            4.299
            + 3.501 * d
            - 2.07 * lage
            + 0.051 * lage ** 2
            + 4.090 * l
            - 1.04 * l * lage
            + 0.01 * f
        )

    def sim_X(self, n, a, l, d, r):
        """
        Simulates patients' statin use X
        ---
        n : int
            number of patients in the dataset
        a : ndarray
            array of patients' ages
        l : ndarray
            array of patients' natural-log transformed low-density lipo-protein
        d : ndarray
            array of patients' incidence of diabetes
        r : ndarray
            array of patients' risk scores

        return ndarray of size n
        """

        pbs = expit(
            -3.471
            + 1.390 * d
            + 0.112 * l
            + 0.973 * (l > np.log(60))
            - 0.046 * (a - 30)
            + 0.003 * (a - 30) ** 2
            + 0.273 * (0.05 <= r) * (r < 0.075)
            + 1.592 * (0.075 <= r) * (r < 0.2)
            + 2.461 * (r >= 0.2)
        )
        return np.array([bernoulli.rvs(pbs[i]) for i in range(n)])

    def sim_Y(self, n, a, l, d, r, x):
        """
        Simulates patients' incidence of atherosclerotic cardiovascular disease Y
        ---
        n : int
            number of patients in the dataset
        a : ndarray
            array of patients' ages
        l : ndarray
            array of patients' natural-log transformed low-density lipo-protein
        d : ndarray
            array of patients' incidence of diabetes
        r : ndarray
            array of patients' risk scores
        x : ndarray
            array of patients' statin use
        """

        pbs = np.zeros((2, n))
        Ya = np.zeros((2, n))

        for i in range(2):
            pbs[i, :] = expit(
                -6.250
                - 0.75 * i
                + 0.35 * i * (5 - l) * (l < np.log(130))
                + 0.45 * np.sqrt(a - 39)
                + 1.75 * d
                + 0.29 * np.exp(r + 1)
                + 0.14 * l ** 2 * (l > np.log(120))
            )
            Ya[i] = np.array([bernoulli.rvs(pbs[i, j]) for j in range(n)])

        return {"Y": x * Ya[1] + (1 - x) * Ya[0], "Y_0": Ya[0]}

    def simulate_patient(self, n, cf=False):
        """
        Generates a dataset of n patients with confounders Z = (A, L, R, D),
        statin use X, and ASCVD incidence Y
        ---
        n : int
            number of patients in the dataset
        cf : bool
             if True, also returns counterfactual Y{a=0}
        """
        v = 0.5 * (55 * uniform.rvs(size=n) + 80)
        # a = sim_age(v)
        a = np.floor(self.sim_age(v))  # patients' ages are assumed to be integers
        l = self.sim_lipo(n, a)
        d = self.sim_diabetes(n, a, l)
        f = self.sim_frailty(n, a)
        r = self.sim_risk(a, l, d, f)

        x = self.sim_X(n, a, l, d, r)

        y_dict = self.sim_Y(n, a, l, d, r, x)
        y = y_dict["Y"]
        y0 = y_dict["Y_0"]

        returns_dict = {
            "data": pd.DataFrame({"X1": a, "X2": l, "X3": r, "X4": d, "A": x, "Y": y})
        }
        if cf:
            returns_dict["Y_0"] = y0
        return returns_dict


if __name__ == "__main__":
    sample_size = 3000  # size of each sample
    # n_sim = 1  # number of repeated simulations

    simulation_dict = PatientDGP().simulate_patient(sample_size, cf=True)
    ascvd_dataset = simulation_dict["data"]
    y0 = simulation_dict["Y_0"]
    ascvd_dataset.to_csv("ascvd_sim_data.csv", index=False)
