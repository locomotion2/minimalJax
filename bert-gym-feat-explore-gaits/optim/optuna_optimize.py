import argparse
import os
import pickle as pkl

import gym
import numpy as np
import optuna
from optuna.samplers import CmaEsSampler, NSGAIISampler, RandomSampler, TPESampler

# from optuna.visualization import plot_optimization_history, plot_param_importances
from rl_zoo3.utils import StoreDict
from stable_baselines3.common.evaluation import evaluate_policy

import custom_envs  # noqa:F401
from custom_envs.constants import RESET_MOTOR_PARAM
from custom_envs.wrappers import MetricWrapper
from optim.common import RandomAgent
from optim.ln_remote import LNRemote

parser = argparse.ArgumentParser()
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
parser.add_argument(
    "--sampler",
    help="Sampler to use when optimizing hyperparameters",
    type=str,
    default="tpe",
    choices=["random", "tpe", "cmaes", "motpe", "nsga"],
)
parser.add_argument(
    "--storage", help="Database storage path if distributed optimization should be used", type=str, default=None
)
parser.add_argument("-name", "--study-name", help="Study name for distributed optimization", type=str, default=None)
parser.add_argument("-local", "--local-params", action="store_true", default=False, help="Use one set of parameters per leg")
parser.add_argument("--n-eval-episodes", help="Number of episode to evaluate a set of parameters", type=int, default=1)
parser.add_argument(
    "-multi", "--multi-objective", action="store_true", default=False, help="Use multi-objective optimization."
)
parser.add_argument(
    "--optimize-energy", action="store_true", default=False, help="Optimize energy/distance cost instead of reward"
)
parser.add_argument("--optimize-speed", action="store_true", default=False, help="Optimize distance/step reward")
parser.add_argument("--optimize-reward", action="store_true", default=False, help="Optimize reward/step")
parser.add_argument("--max-episode-steps", help="Overrride the timeout", type=int)
parser.add_argument(
    "--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor"
)
args = parser.parse_args()


N_TRIALS = 1000
N_STARTUP_TRIALS = 5
N_EVAL_EPISODES = args.n_eval_episodes  # More episodes account for eval noise
SEED = 42
MIN_DISTANCE_TRAVELLED = 0.1  # 10 cm
MIN_EPISODE_LENGTH = 120  # 2s at 60 Hz


# Restart observers
ln_remote = LNRemote(os.environ["LN_MANAGER"])
ln_remote.restart_observers()
# Stop gamepad controller to free the topic
ln_remote.stop_process("all/applications/cpg_gamepad")
ln_remote.stop_process("all/applications/gamepad_control")
# Start tracking if not already done
ln_remote.start_process("all/tracking_system/dtrack_ln_publisher_0")

env_kwargs = args.env_kwargs or {}

env = gym.make(args.env, one_param_per_leg=args.local_params, **env_kwargs)
# Update max episode steps
if args.max_episode_steps is not None:
    env._max_episode_steps = args.max_episode_steps

random_agent = RandomAgent(env)

# TODO: feed good initial parameters
study_kwargs = {}
if args.multi_objective:
    sampler = {
        "tpe": TPESampler(n_startup_trials=N_STARTUP_TRIALS, multivariate=True, seed=SEED),
        "nsga": NSGAIISampler(population_size=50, seed=SEED),
        "random": RandomSampler(seed=SEED),
    }[args.sampler]
    # Maximise reward, minimize energy cost
    study_kwargs["directions"] = ["maximize", "minimize"]
else:
    sampler = {
        "tpe": TPESampler(n_startup_trials=N_STARTUP_TRIALS, multivariate=True, seed=SEED),
        "cmaes": CmaEsSampler(seed=SEED),
        "random": RandomSampler(seed=SEED),
    }[args.sampler]
    study_kwargs["direction"] = "minimize" if args.optimize_energy else "maximize"

study = optuna.create_study(
    sampler=sampler,
    storage=args.storage,
    study_name=args.study_name,
    load_if_exists=True,
    **study_kwargs,
)

# Wrap to compute metrics
env = MetricWrapper(env)

# TODO: save default motor params
study.RESET_MOTOR_PARAM = RESET_MOTOR_PARAM
n_valid = 0

if "ExploreCPGGait" in args.env:
    # Enqueue trot
    trot_params = {"front_right": 0.5, "front_left": 0, "rear_right": 0, "rear_left": 0.5}
    # if "WithParams" in args.env:
    #     trot_params.update({"omega_swing": 7, "omega_stance": 4})
    study.enqueue_trial(trot_params, skip_if_exists=False)
    # Enqueue walk
    walk_params = {"front_right": 0.5, "front_left": 0, "rear_right": 0.75, "rear_left": 0.25}
    # if "WithParams" in args.env:
    #     walk_params.update({"omega_swing": 7, "omega_stance": 4})
    study.enqueue_trial(walk_params, skip_if_exists=False)

try:
    for _ in range(1, N_TRIALS + 1):

        trial = study.ask()
        env.reset_metrics()
        env.sample_params(trial)
        episode_rewards, episode_lengths = evaluate_policy(
            random_agent,
            env,
            N_EVAL_EPISODES,
            warn=False,
            callback=None,
            return_episode_rewards=True,
        )
        episode_rewards, episode_lengths = np.array(episode_rewards), np.array(episode_lengths)

        trial.set_user_attr("mean_energy_cost", env.mean_energy_cost)
        trial.set_user_attr("mean_speed", env.mean_speed)
        trial.set_user_attr("mean_distance", env.mean_distance_travelled)
        trial.set_user_attr("mean_dx", env.mean_distance_travelled)
        trial.set_user_attr("mean_dy", env.mean_dy)
        trial.set_user_attr("mean_deviation", env.mean_heading_deviation)
        # Convert to float to be serializable
        trial.set_user_attr("total_steps", float(np.sum(episode_lengths)))

        if args.multi_objective:
            study.tell(trial, (env.mean_speed, env.mean_energy_cost))
        else:
            if args.optimize_energy:
                if env.mean_distance_travelled < MIN_DISTANCE_TRAVELLED:
                    study.tell(trial, state=optuna.trial.TrialState.PRUNED)  # tell the pruned state
                else:
                    study.tell(trial, env.mean_energy_cost)
                    n_valid += 1
            elif args.optimize_reward:
                if np.mean(episode_lengths) < MIN_EPISODE_LENGTH:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                else:
                    study.tell(trial, np.mean(episode_rewards / episode_lengths))
                    n_valid += 1
            else:
                n_valid += 1
                study.tell(trial, env.mean_speed)
                # Optimize distance travelled
                # study.tell(trial, env.mean_distance_travelled)

        if args.multi_objective:
            print(
                f"Trial {trial.number} - mean_speed = {env.mean_speed:.3f} m/s "
                "- mean_energy_cost = {env.mean_energy_cost:.2f} "
                f"mean_distance = { env.mean_distance_travelled * 100:.2f} cm"
            )
        else:
            best_value = 0.0
            if n_valid > 0:
                best_value = study.best_trial.value
            if args.optimize_reward:
                print(
                    f"Trial {trial.number} - "
                    f"Mean reward/step = {np.mean(episode_rewards / episode_lengths):.2f} - Best = {best_value:.4f}"
                )
            else:
                print(
                    f"Trial {trial.number} -  mean_speed = {env.mean_speed:.3f} m/s "
                    f"- mean_energy_cost = {env.mean_energy_cost:.2f} "
                    f"- mean_distance = { env.mean_distance_travelled * 100:.2f} cm - Best = {best_value:.4f}"
                )

except KeyboardInterrupt:
    pass

# Reset motor position
env.unwrapped.ln_server.bert_controller.set_motor_positions(RESET_MOTOR_PARAM)

print("Number of finished trials: ", len(study.trials))

if args.multi_objective:
    print("Pareto front:")

    trials = sorted(study.best_trials, key=lambda t: t.values)

    for trial in trials:
        print()
        print(f"  Trial#{trial.number}")
        print(f"  Values: mean_speed={trial.values[0]}, energy_cost={trial.values[1]}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value:.5f}")

else:
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value:.3f}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value:.5f}")

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print(f"    {key}: {value:.2f}")

# Write report
study.trials_dataframe().to_csv("logs/study_results.csv")

with open("logs/study.pkl", "wb+") as f:
    pkl.dump(study, f)

# if not args.multi_objective:
#
#     fig1 = plot_optimization_history(study)
#     fig2 = plot_param_importances(study)
#
#     fig1.show()
#     fig2.show()
