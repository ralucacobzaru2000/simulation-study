from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ascvd_dgp import PatientDGP
from master_model import MasterModel
from multithreading_utils import MultithreadingUtils as MUtils

from confounders import add_variables


def simulate_dataset(
    config_file: str, sample_size: int, extra_vars: int, extra_params: float,
) -> Dict[str, float]:
    sim_dict = PatientDGP().simulate_patient(sample_size, cf=True)
    dataset = sim_dict["data"]
    var_types = ["noise", "exposure", "outcome", "M-bias"]
    for i in range(8):
        if extra_vars[i] > 0:
            dataset = add_variables(
                dataset,
                cf_var=sim_dict["Y_0"],
                n_vars=extra_vars[i],
                type=var_types[i // 2],
                strong_assoc=bool(i % 2),
                large_noise=bool(i % 2),
                prop_bool=0.2,
                extra_params=extra_params[i // 2],
            )
    master_model = MasterModel(config_file, sample_size)
    results = master_model.process_dataset(dataset)

    dataset_results = {}

    # for j in range(len(master_model.models)):
    for j in range(len(results)):
        conf_low_e = results[j][PatientSim.CONF_LOW_E]
        conf_high_e = results[j][PatientSim.CONF_HIGH_E]
        sim_results = {
            PatientSim.EFFECTS: results[j][PatientSim.MEAN_E],
            PatientSim.STD_EFF: results[j][PatientSim.STD_E],
            PatientSim.CLDS: conf_high_e - conf_low_e,
            PatientSim.COVERS: (
                conf_low_e < PatientSim.TRUE_EFFECT
                and conf_high_e > PatientSim.TRUE_EFFECT
            ),
        }

        if len(results) == len(master_model.models):
            model = master_model.models[j]
        else:
            model = master_model.models[j // 2]
        current_method = model.get_method()
        current_learner = str(model.get_learner_string())
        current_std = results[j][PatientSim.STD_ERROR]

        dataset_current_learner = dataset_results.get(current_method, {})
        dataset_current_std = dataset_current_learner.get(current_learner, {})

        dataset_current_std[current_std] = sim_results
        dataset_current_learner[current_learner] = dataset_current_std
        dataset_results[current_method] = dataset_current_learner

    return dataset_results


class PatientSim:
    TRUE_EFFECT: float = -0.1081508

    STD_ERROR: str = "std error"
    ERROR: str = "error type"

    MEAN_E: str = "diff mean"
    STD_E: str = "diff std"
    CONF_LOW_E: str = "diff lower"
    CONF_HIGH_E: str = "diff upper"

    EFFECTS: str = "effects"
    STD_EFF: str = "stds effect"
    CLDS: str = "clds"
    COVERS: str = "covers"

    def __init__(
        self,
        config_file: str,
        n_datasets: int,
        sample_size: int,
        extra_vars: int,
        extra_params: float,
    ) -> None:
        self.config_file = config_file
        self.mutils = MUtils()
        self.datasets_results = self.__simulate(
            n_datasets, sample_size, extra_vars, extra_params,
        )

    def __get_result_dict(self) -> Dict[str, List[float]]:
        return {
            self.EFFECTS: [],
            self.STD_EFF: [],
            self.CLDS: [],
            self.COVERS: [],
        }

    def __simulate(
        self, n_datasets: int, sample_size: int, extra_vars: int, extra_params: float,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        datasets_results = {}

        master_model = MasterModel(self.config_file, sample_size)
        self.learners_IPW_len = len(master_model.learners_IPW)
        self.learners_standardization_len = len(master_model.learners_standardization)

        args_list = [
            (self.config_file, sample_size, extra_vars, extra_params)
            for i in range(n_datasets)
        ]
        # comment out (and comment in next part) to deactivate multiprocessing
        pool_results = self.mutils.run_function(simulate_dataset, args_list, True)

        # start of sequential result computation
        """
        pool_results = []
        pool_index = 1
        # error_index = 1

        for args in args_list:
            print("Dataset", pool_index)
            pool_index += 1
            while True:
                try:
                    result = simulate_dataset(*args)
                    pool_results.append(result)
                    break
                except Exception as e:
                    print(type(e))
                    print(e)
                    print(e.__doc__)
        """
        # end of sequential results

        for j in range(len(master_model.models)):
            model = master_model.models[j]
            current_method = model.get_method()
            current_learner = str(model.get_learner_string())

            for current_std in master_model.std_types:
                if current_std != "comment":
                    if not master_model.__getattribute__(current_std):
                        continue
                    dataset_current_method = datasets_results.get(current_method, {})
                    dataset_current_learner = dataset_current_method.get(
                        current_learner, {}
                    )
                    dataset_result = dataset_current_learner.get(
                        current_std, self.__get_result_dict()
                    )

                    for key in dataset_result:
                        for pool_result in pool_results:
                            dataset_result[key].append(
                                pool_result[current_method][current_learner][
                                    current_std
                                ][key]
                            )

                    dataset_current_learner[current_std] = dataset_result
                    dataset_current_method[current_learner] = dataset_current_learner
                    datasets_results[current_method] = dataset_current_method

        for model in datasets_results:
            for learner in datasets_results[model]:
                for std_type in datasets_results[model][learner]:
                    data = datasets_results[model][learner][std_type]
                    data["bias"] = np.mean(data["effects"]) - self.TRUE_EFFECT
                    data["ase"] = np.mean(data["stds effect"])
                    data["ese"] = np.std(
                        np.array(data["effects"]) - self.TRUE_EFFECT, ddof=1
                    )
                    data["cld"] = np.mean(data["clds"])
                    data["cover"] = np.mean(data["covers"])

                    plt.figure()
                    plt.hist(
                        np.array(data["effects"]) - self.TRUE_EFFECT, 50, (-0.15, 0.15)
                    )
                    plt.title("est ACE - true ACE: %s" % learner)
                    plt.savefig("hist_%s_ace.png" % learner)

                    plt.figure()
                    plt.hist(
                        np.array(data["effects"])
                        - 1.96 * np.array(data["stds effect"]),
                        50,
                    )
                    plt.title("CI low: %s" % learner)
                    plt.savefig("hist_%s_CI_low" % learner)

                    plt.figure()
                    plt.hist(
                        np.array(data["effects"])
                        + 1.96 * np.array(data["stds effect"]),
                        50,
                    )
                    plt.title("CI high: %s" % learner)
                    plt.savefig("hist_%s_CI_high" % learner)

                    del data["effects"]
                    del data["stds effect"]
                    del data["clds"]
                    del data["covers"]

        return datasets_results

    def export_csv(self, file_dest: str) -> None:
        results_all = pd.DataFrame()
        for model_key in self.datasets_results:
            for learner_key in self.datasets_results[model_key]:
                results = pd.DataFrame(self.datasets_results[model_key][learner_key])
                results = results.T
                results.insert(0, "model", [model_key] * len(results.index))
                results.insert(1, "learner", [learner_key] * len(results.index))
                results_all = results_all.append(results)
        results_all.reset_index()
        results_all.index.name = "std types"
        results_all.to_csv(file_dest)


if __name__ == "__main__":
    import time
    import datetime

    n_datasets = 2000
    sample_size = 3000

    extra_params = [
        [0.5, 0.5, 0.5, 2],
        [0.65, 1.55, 2, 0.5],
        [0.65, 1.55, 2, 0.5],
        [0.6, 1.5, 2, 0.5, 0.1, 0.5, 2, 0.5, 0.7, 1.65, 1.75, 0.8],
    ]

    t0 = time.time()
    extra_vars = [1, 0, 2, 1, 2, 1, 2, 1]
    simulation = PatientSim(
        config_file="dr_config.json",
        n_datasets=n_datasets,
        sample_size=sample_size,
        extra_vars=extra_vars,
        extra_params=extra_params,
    )
    t1 = time.time()
    simulation.export_csv("mix_test.csv")
    print("\nTotal time: ", str(datetime.timedelta(seconds=t1 - t0)))
