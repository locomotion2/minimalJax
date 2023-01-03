import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from optuna.visualization._optimization_history import _get_optimization_history_info_list

from optim.study_utils import load_study

# Do not use type3 fonts
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# Activate seaborn
seaborn.set()
# Seaborn style
seaborn.set(style="whitegrid")


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
    parser.add_argument("-latex", "--latex", help="Enable latex support", action="store_true", default=False)
    parser.add_argument("--figsize", help="Figure size, width, height in inches.", nargs=2, type=int, default=[7.9, 5.4])
    parser.add_argument("--fontsize", help="Font size", type=int, default=16)
    args = parser.parse_args()

    # Enable LaTeX support
    if args.latex:
        plt.rc("text", usetex=True)

    study = load_study(args)

    info_list = _get_optimization_history_info_list(study, target=None, target_name="Objective Value", error_bar=False)
    target_name = "Objective Value"
    plt.figure(f"Optimization History Plot - {args.study_name}", figsize=args.figsize)

    if "pronk" in args.study_name:
        plt.title("Optimization History (pronking)", fontsize=args.fontsize)
    else:
        plt.title("Optimization History (trotting)", fontsize=args.fontsize)

    plt.xlabel("Trial", fontsize=args.fontsize)

    y_factor = 1
    if "pronk" in args.study_name:
        # Convert mean reward / step to episodic reward
        y_factor = 300

    if "pronk" in args.study_name:
        plt.ylabel("Episodic Reward", fontsize=args.fontsize)
    else:
        plt.ylabel("Mean Speed (m/s)", fontsize=args.fontsize)

    cmap = plt.get_cmap("tab10")  # Use tab10 colormap for similar outputs to plotly.

    for i, (trial_numbers, values_info, best_values_info) in enumerate(info_list):
        plt.scatter(
            x=trial_numbers,
            y=np.array(values_info.values) * y_factor,
            color=cmap(0) if len(info_list) == 1 else cmap(2 * i),
            alpha=1,
            label="Trial",
            # s=30,
        )

        if best_values_info is not None:
            # Deduplicate
            best_trial_numbers, best_values = [], []
            for idx, value in enumerate(best_values_info.values):
                if value not in best_values:
                    # Make it flat until new trial
                    if len(best_trial_numbers) > 0 and idx - 1 not in best_trial_numbers:
                        best_trial_numbers.append(idx - 1)
                        best_values.append(best_values[-1])

                    best_trial_numbers.append(idx)
                    best_values.append(value)
            # Duplicate last entry
            best_trial_numbers.append(idx)
            best_values.append(best_values[-1])

            plt.plot(
                best_trial_numbers,
                np.array(best_values) * y_factor,
                marker="o",
                color=cmap(3) if len(info_list) == 1 else cmap(2 * i + 1),
                alpha=0.8,
                label=best_values_info.label_name,
                linewidth=3,
            )
    # Hand-tuned baseline
    handtuned_value = 1200 if "pronk" in args.study_name else 0.15
    plt.axhline(
        y=handtuned_value,
        color=(0, 0, 0),
        linestyle="--",
        linewidth=4,
        alpha=0.8,
        label="Hand-tuned parameters",
    )

    # 10 minutes
    plt.axvline(
        x=60,
        color="#54B337",
        linestyle="-.",
        linewidth=5,
        alpha=0.5,
    )
    if "pronk" in args.study_name:
        # Pronking
        text = plt.text(x=52, y=700, s="10 minutes", rotation=90, color="#54B337", verticalalignment="center")
    else:
        # Trotting
        text = plt.text(x=49, y=0.25, s="10 minutes", rotation=90, color="#54B337", verticalalignment="center")

    # White bg
    text.set_bbox(dict(facecolor="white", alpha=0.9, edgecolor="white"))

    # 30 minutes
    # plt.axvline(
    #     x=180,
    #     color="#54B337",
    #     linestyle="-.",
    #     linewidth=5,
    #     alpha=0.5,
    # )
    # plt.text(x=172, y=0.20, s="30 minutes", rotation=90, color="#54B337", verticalalignment='center')

    plt.legend(loc="lower right", framealpha=0.9, fontsize=args.fontsize)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

    # fig = optuna.visualization.plot_optimization_history(study)
    # fig.show()
