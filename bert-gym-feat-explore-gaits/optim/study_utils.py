import pickle
import re
from typing import Dict, List

import numpy as np
import optuna
from cpg_gamepad.cpg import skew_sim_matrix
from optuna.trial import FrozenTrial, TrialState

from custom_envs.cpg.plot import compute_gait_stats

PLOT_PUBLISHER_PORT = 5559

coupling_params_pattern = re.compile(r"((front)|(rear))_((left)|(right))")
gait_params_pattern = re.compile(r"((omega)|(ground)|(desired_step_len))")


def get_coupling_matrix(trial: FrozenTrial) -> np.ndarray:
    coupling_params = {f"{key}": trial.params[key] for key in trial.params if coupling_params_pattern.match(key)}
    return 2 * np.pi * skew_sim_matrix(**coupling_params)


def get_values(trials: List[FrozenTrial], key: str) -> np.ndarray:
    return np.array([trial.user_attrs[key] for trial in trials])


def get_gait_stats(trial: FrozenTrial) -> Dict[str, float]:
    return compute_gait_stats(
        np.pi * trial.params["omega_swing"],
        np.pi * trial.params["omega_stance"],
        get_coupling_matrix(trial),
        print_info=False,
    )


def gait_stats_to_dict(gait_stats: List[Dict[str, float]]) -> Dict[str, np.ndarray]:
    stats = {key: np.zeros(len(gait_stats)) for key in gait_stats[0].keys()}
    for key in stats.keys():
        for gait_idx, gait_dict in enumerate(gait_stats):
            stats[key][gait_idx] = gait_dict[key]
    return stats


def get_gait_values(trials: List[FrozenTrial], key: str) -> np.ndarray:
    return np.array(
        [
            compute_gait_stats(
                np.pi * trial.params["omega_swing"],
                np.pi * trial.params["omega_stance"],
                print_info=False,
            )[key]
            for trial in trials
        ]
    )


def load_study(args) -> optuna.Study:
    if args.study_name is None:
        assert args.study_file is not None, "No --study-file, nor --study-name were provided."
        with open(args.study_file, "rb") as f:
            study = pickle.load(f)

    else:
        assert args.storage is not None, "No storage was specified."

        study = optuna.load_study(
            study_name=args.study_name,
            storage=args.storage,
        )
    return study


def get_completed_trials(args) -> List[FrozenTrial]:
    study = load_study(args)
    trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    return trials
