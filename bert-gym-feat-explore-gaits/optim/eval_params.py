import argparse
import json
import os
from typing import Dict

import gym
import numpy as np
import optuna
import yaml
from cpg_gamepad.cpg import skew_sim_matrix
from rl_zoo3.utils import StoreDict
from stable_baselines3.common.evaluation import evaluate_policy

import custom_envs  # noqa:F401
from custom_envs.constants import RESET_MOTOR_PARAM, TRACKING_DISABLED
from custom_envs.wrappers import MetricWrapper
from optim.common import PlotFootTrajectoryCallback, RandomAgent
from optim.ln_remote import LNRemote


def set_bert_env_parameters(bert_env: gym.Env, parameters: Dict[str, float]):

    valid_params = False
    if "front_right" in parameters:
        leg_phase_shifts = {key: parameters[key] for key in ["front_right", "front_left", "rear_right", "rear_left"]}
        bert_env.coupling_matrix = 2 * np.pi * skew_sim_matrix(**leg_phase_shifts)
        valid_params = True

    if "omega_swing" in parameters:
        # CPG env
        bert_env.omega_swing = parameters["omega_swing"]
        bert_env.omega_stance = parameters["omega_stance"]
        if "desired_step_len" in parameters:
            bert_env.desired_step_len = parameters["desired_step_len"]
            # Do not optimize ground clearance when pronking
            bert_env.ground_clearance = parameters["ground_clearance"]
            bert_env.ground_penetration = parameters["ground_penetration"]
    elif "radius_offset_0" in parameters:
        # One parameter per leg, reflex controller
        for leg_idx in range(bert_env.num_legs):
            bert_env.state_machine.radius_offset[leg_idx] = parameters[f"radius_offset_{leg_idx}"]
            bert_env.state_machine.bang_bang_thresholds[leg_idx] = parameters[f"threshold_{leg_idx}"]
            angle_amplitude = parameters[f"angle_amplitude_{leg_idx}"]
            bert_env.state_machine.bang_bang_amplitude_angles[leg_idx] = np.deg2rad(angle_amplitude)
            bert_env.state_machine.bang_bang_amplitude_radii[leg_idx] = parameters[f"radius_amplitude_{leg_idx}"]

    elif "radius_offset" in parameters:
        # reflex controller, one set of parameters
        bert_env.state_machine.radius_offset = parameters["radius_offset"] * np.ones((bert_env.state_machine.num_legs,))
        bert_env.state_machine.bang_bang_thresholds = parameters["threshold"] * np.ones((bert_env.state_machine.num_legs,))
        angle_amplitude = parameters["angle_amplitude"] * np.ones((bert_env.state_machine.num_legs,))
        bert_env.state_machine.bang_bang_amplitude_angles = np.deg2rad(angle_amplitude)
        bert_env.state_machine.bang_bang_amplitude_radii = parameters["radius_amplitude"] * np.ones((bert_env.num_legs,))

    else:
        if not valid_params:
            raise ValueError("Parameters not recognized")


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-file", help="Path to a json file that contains parameters values", type=str)
parser.add_argument(
    "--env",
    type=str,
    help="Env id",
    choices=[
        "ModesBertReal-v1",
        "ModesBertSim-v1",
        "CPGBertReal-v1",
        "CPGWalkBertReal-v1",
        "CPGBertPronk-v1",
        "ExploreCPGGait-v0",
        "ExploreCPGGaitWithParams-v0",
    ],
    required=True,
)
parser.add_argument("-name", "--study-name", help="Study name used during hyperparameter optimization", type=str)
parser.add_argument("--storage", help="Database storage path used during hyperparameter optimization", type=str)
parser.add_argument("-trial", "--trial-id", help="Id of the trial to retrieve", type=int)
parser.add_argument("--n-eval-episodes", help="Number of episode to evaluate a set of parameters", type=int, default=1)
parser.add_argument("--max-episode-steps", help="Overrride the timeout", type=int)
parser.add_argument("-plot", "--plot", action="store_true", default=False, help="Plot x/z diagram.")
parser.add_argument("-b", "--backward", action="store_true", default=False, help="Go backward")
parser.add_argument("--json", action="store_true", default=False, help="Output json instead of yaml")
parser.add_argument(
    "--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor"
)
args = parser.parse_args()

if args.study_name is None:
    assert args.input_file is not None, "No input file specified."
    with open(args.input_file, "r") as f:
        parameters = json.load(f)
else:
    assert args.storage is not None, "No storage was specified."
    # kwargs = dict(directions=["maximize", "minimize"]) if args.multi else  dict(direction="maximize")
    study = optuna.load_study(
        study_name=args.study_name,
        storage=args.storage,
    )
    parameters = study.trials[args.trial_id].params

if args.json:
    print(json.dumps(parameters))
else:
    print(yaml.dump(parameters))

env_kwargs = args.env_kwargs or {}

ln_remote = LNRemote(os.environ["LN_MANAGER"])
# Stop gamepad controller to free the topic
ln_remote.stop_process("all/applications/cpg_gamepad")
ln_remote.stop_process("all/applications/gamepad_control")

env = gym.make(args.env, backward=args.backward, **env_kwargs)
# Update max episode steps
if args.max_episode_steps is not None:
    env._max_episode_steps = args.max_episode_steps

random_agent = RandomAgent(env)

# unwrap
bert_env = env.unwrapped

set_bert_env_parameters(bert_env, parameters)

# Wrap to compute metrics
if not TRACKING_DISABLED:
    env = MetricWrapper(env)
    # Restart observers
    ln_remote.restart_observers()

plot_callback = PlotFootTrajectoryCallback(env, leg_idx=0)

callback = None
if args.plot:
    callback = plot_callback


try:
    episode_rewards, episode_lengths = evaluate_policy(
        random_agent,
        env,
        args.n_eval_episodes,
        warn=False,
        callback=callback,
        return_episode_rewards=True,
    )
    mean_reward = np.mean(episode_rewards)
    reward_std = np.std(episode_rewards)
    total_length = np.sum(episode_lengths)
    mean_reward_per_step = np.sum(episode_rewards) / total_length

    print(f"Reward = {mean_reward:.3f} +/- {reward_std:.3f} - {total_length} steps - Reward/step = {mean_reward_per_step:.4f}")
    # Print energy cost and mean speed
    if not TRACKING_DISABLED:
        env.print_metrics()
except KeyboardInterrupt:
    pass

# Reset motor position
bert_env.ln_server.bert_controller.set_motor_positions(RESET_MOTOR_PARAM)

if args.plot:
    plot_callback.plot()
