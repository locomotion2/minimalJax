import argparse
import json
from typing import Any, Dict, List, Optional, Sequence, Union

from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial, TrialState
from optuna.visualization._pareto_front import _get_non_pareto_front_trials, _make_json_compatible
from optuna.visualization._plotly_imports import _imports

from optim.study_utils import load_study

if _imports.is_successful():
    from optuna.visualization._plotly_imports import go


def _make_hovertext(trial: FrozenTrial) -> str:
    user_attrs = {key: _make_json_compatible(value) for key, value in trial.user_attrs.items()}
    user_attrs_dict = {"user_attrs": user_attrs} if user_attrs else {}
    text = json.dumps(
        {
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            **user_attrs_dict,
        },
        indent=2,
    )
    return text.replace("\n", "<br>")


def _normalize_value(value: Optional[float], direction: StudyDirection) -> float:
    if value is None:
        value = float("inf")

    if direction is StudyDirection.MAXIMIZE:
        value = -value

    return value


def _make_marker(
    trials: Sequence[FrozenTrial],
    include_dominated_trials: bool,
    dominated_trials: bool = False,
    infeasible: bool = False,
) -> Dict[str, Any]:
    if dominated_trials and not include_dominated_trials:
        assert len(trials) == 0

    if infeasible:
        return {
            "color": "#cccccc",
        }
    elif dominated_trials:
        return {
            "line": {"width": 0.5, "color": "Grey"},
            "color": [t.number for t in trials],
            "colorscale": "Blues",
            "colorbar": {
                "title": "#Trials",
            },
        }
    else:
        return {
            "line": {"width": 0.5, "color": "Grey"},
            "color": [t.number for t in trials],
            "colorscale": "Reds",
            "colorbar": {
                "title": "#Best trials",
                "x": 1.1 if include_dominated_trials else 1,
                "xpad": 40,
            },
        }


def _make_scatter_object_base(
    n_dim: int,
    trials: Sequence[FrozenTrial],
    key,
    axis_order: List[int],
    include_dominated_trials: bool,
    hovertemplate: str,
    infeasible: bool = False,
    dominated_trials: bool = False,
) -> Union["go.Scatter", "go.Scatter3d"]:
    assert n_dim in (2, 3)
    marker = _make_marker(
        trials,
        include_dominated_trials,
        dominated_trials=dominated_trials,
        infeasible=infeasible,
    )
    if n_dim == 2:
        return go.Scatter(
            x=[t.value for t in trials],
            y=[t.user_attrs[key] for t in trials],
            text=[_make_hovertext(t) for t in trials],
            mode="markers",
            hovertemplate=hovertemplate,
            marker=marker,
            showlegend=False,
        )
    else:
        raise NotImplementedError()


def plot_pareto_front(
    study,
    best_trials,
    key,
    directions,
    *,
    target_names: Optional[List[str]] = None,
    include_dominated_trials: bool = True,
    axis_order: Optional[List[int]] = None,
) -> "go.Figure":
    _imports.check()

    n_dim = len(directions)
    if n_dim not in (2, 3):
        raise ValueError("`plot_pareto_front` function only supports 2 or 3 objective studies.")

    if target_names is None:
        target_names = [f"Objective {i}" for i in range(n_dim)]
    elif len(target_names) != n_dim:
        raise ValueError(f"The length of `target_names` is supposed to be {n_dim}.")

    non_best_trials = []
    if include_dominated_trials:
        non_best_trials = _get_non_pareto_front_trials(study.trials, best_trials)

    infeasible_trials = []

    if axis_order is None:
        axis_order = list(range(n_dim))
    else:
        if len(axis_order) != n_dim:
            raise ValueError(f"Size of `axis_order` {axis_order}. Expect: {n_dim}, Actual: {len(axis_order)}.")
        if len(set(axis_order)) != n_dim:
            raise ValueError(f"Elements of given `axis_order` {axis_order} are not unique!.")
        if max(axis_order) > n_dim - 1:
            raise ValueError(
                f"Given `axis_order` {axis_order} contains invalid index {max(axis_order)} " f"higher than {n_dim - 1}."
            )
        if min(axis_order) < 0:
            raise ValueError(f"Given `axis_order` {axis_order} contains invalid index {min(axis_order)} " "lower than 0.")

    def _make_scatter_object(
        trials: Sequence[FrozenTrial],
        hovertemplate: str,
        infeasible: bool = False,
        dominated_trials: bool = False,
    ) -> Union["go.Scatter", "go.Scatter3d"]:
        return _make_scatter_object_base(
            n_dim,
            trials,
            key,
            axis_order,  # type: ignore
            include_dominated_trials,
            hovertemplate=hovertemplate,
            infeasible=infeasible,
            dominated_trials=dominated_trials,
        )

    data = [
        _make_scatter_object(
            infeasible_trials,
            hovertemplate="%{text}<extra>Infeasible Trial</extra>",
            infeasible=True,
        ),
        _make_scatter_object(
            non_best_trials,
            hovertemplate="%{text}<extra>Feasible Trial</extra>",
            dominated_trials=True,
        ),
        _make_scatter_object(
            best_trials,
            hovertemplate="%{text}<extra>Best Trial</extra>",
            dominated_trials=False,
        ),
    ]

    if n_dim == 2:
        layout = go.Layout(
            title="Pareto-front Plot",
            xaxis_title=target_names[axis_order[0]],
            yaxis_title=target_names[axis_order[1]],
        )
    else:
        layout = go.Layout(
            title="Pareto-front Plot",
            scene={
                "xaxis_title": target_names[axis_order[0]],
                "yaxis_title": target_names[axis_order[1]],
                "zaxis_title": target_names[axis_order[2]],
            },
        )
    return go.Figure(data=data, layout=layout)


def _dominates(trial0: FrozenTrial, trial1: FrozenTrial, key, directions: Sequence[StudyDirection]) -> bool:
    values0 = trial0.value, trial0.user_attrs[key]
    values1 = trial1.value, trial1.user_attrs[key]

    assert values0 is not None
    assert values1 is not None

    if len(values0) != len(values1):
        raise ValueError("Trials with different numbers of objectives cannot be compared.")

    if len(values0) != len(directions):
        raise ValueError("The number of the values and the number of the objectives are mismatched.")

    if trial0.state != TrialState.COMPLETE:
        return False

    if trial1.state != TrialState.COMPLETE:
        return True

    normalized_values0 = [_normalize_value(v, d) for v, d in zip(values0, directions)]
    normalized_values1 = [_normalize_value(v, d) for v, d in zip(values1, directions)]

    if normalized_values0 == normalized_values1:
        return False

    return all(v0 <= v1 for v0, v1 in zip(normalized_values0, normalized_values1))


def get_pareto_front_trials_2d(trials, key, directions) -> List:
    trials = [trial for trial in trials if trial.state == TrialState.COMPLETE]

    n_trials = len(trials)
    if n_trials == 0:
        return []

    trials.sort(
        key=lambda trial: (
            _normalize_value(trial.value, directions[0]),
            _normalize_value(trial.user_attrs[key], directions[1]),
        ),
    )

    last_nondominated_trial = trials[0]
    pareto_front = [last_nondominated_trial]
    for i in range(1, n_trials):
        trial = trials[i]
        if _dominates(last_nondominated_trial, trial, key, directions):
            continue
        pareto_front.append(trial)
        last_nondominated_trial = trial

    pareto_front.sort(key=lambda trial: trial.number)
    return pareto_front


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--study-file", help="Path to a pickle file contained a saved study", type=str)
parser.add_argument("-name", "--study-name", help="Study name used during hyperparameter optimization", type=str)
parser.add_argument("--storage", help="Database storage path used during hyperparameter optimization", type=str)
parser.add_argument("--key", help="Secondary objective", default="speed", choices=["speed", "energy"], type=str)
args = parser.parse_args()

study = load_study(args)

# Hack to set the best trials
# it will use value and user_attrs[key]
# TODO: use only user_attrs[key]
key = "mean_speed" if args.key == "speed" else "mean_energy_cost"

if key == "mean_energy_cost":
    target_names = ["mean_speed", "mean_energy_cost"]
    directions = [StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE]
elif key == "mean_speed":
    target_names = ["mean_energy_cost", "mean_speed"]
    directions = [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE]
best_trials = get_pareto_front_trials_2d(study.trials, key, directions)

fig = plot_pareto_front(study, best_trials, key, directions, target_names=target_names, include_dominated_trials=True)
fig.show()
