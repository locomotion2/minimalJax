import argparse
import subprocess

from optim.study_utils import coupling_params_pattern, gait_params_pattern, load_study

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--study-file", help="Path to a pickle file contained a saved study", type=str)
    parser.add_argument("-name", "--study-name", help="Study name used during hyperparameter optimization", type=str)
    parser.add_argument(
        "--storage",
        help="Database storage path used during hyperparameter optimization",
        type=str,
        default="sqlite:///logs/studies.db",
    )
    parser.add_argument("-trial", "--trial-id", help="Id of the trial to retrieve", type=int)

    parser.add_argument("--args", help="Arguments to pass to the plotting script", type=str)
    args = parser.parse_args()

    study = load_study(args)

    parameters = study.trials[args.trial_id].params
    coupling_params = ["-shift"] + [f"{key}:{parameters[key]:.2f}" for key in parameters if coupling_params_pattern.match(key)]
    cpg_params = ["-param"] + [f"{key}:{parameters[key]:.2f}" for key in parameters if gait_params_pattern.match(key)]
    cmd_args = [] if args.args is None else args.args.split(" ")
    python_cmd = ["python", "custom_envs/cpg/plot.py"] + coupling_params + cpg_params + cmd_args
    print(" ".join(python_cmd))
    subprocess.call(python_cmd)
