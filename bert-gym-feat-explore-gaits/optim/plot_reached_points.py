import argparse
import subprocess
import time
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import zmq
from optuna.trial import FrozenTrial

from optim.study_utils import PLOT_PUBLISHER_PORT, gait_stats_to_dict, get_completed_trials, get_gait_stats, get_values

# Do not use type3 fonts
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# Activate seaborn
seaborn.set()
# Seaborn style
seaborn.set(style="whitegrid")


def create_annotation(
    scatter: matplotlib.collections.PathCollection,
    trials: List[FrozenTrial],
    args: argparse.Namespace,
):
    annotation = plt.annotate(
        "",
        xy=(0, 0),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc=seaborn.color_palette("Paired")[0], alpha=0.8),
    )
    annotation.set_visible(False)

    fig, axes = plt.gcf(), plt.gca()

    def on_click(event):
        if event.inaxes == axes:
            is_contained, annotation_index = scatter.contains(event)
            if is_contained:
                idx = annotation_index["ind"][0]
                annotation.xy = scatter.get_offsets()[idx]
                trial = trials[idx]
                infos = "\n".join([f"{key}: {trial.params[key]:.2f}" for key in trial.params])
                # infos += "\n".join([f"{key}: {trial.user_attrs[key]:.2f}" for key in ["mean_speed"]])
                infos += f"\nspeed: {trial.user_attrs['mean_speed']:.2f}m/s \ndev:{trial.user_attrs['mean_deviation']:.2f}deg"

                # Double click: plot gait
                if event.dblclick:
                    if args.sim_replay:
                        context = zmq.Context()
                        publisher = context.socket(zmq.PUSH)
                        publisher.connect(f"tcp://localhost:{PLOT_PUBLISHER_PORT}")
                        publisher.LINGER = 0  # non blocking
                        publisher.send_pyobj(trial)
                        # Wait a little bit before closing the socket
                        time.sleep(0.1)
                        publisher.close()
                        context.term()
                    else:
                        subprocess.call(
                            ["python", "optim/plot_gait.py", "-name", args.study_name, "-trial", str(trial.number)]
                        )

                annotation.set_text(f"Trial {trial.number}\n" + infos)
                annotation.set_visible(True)
                fig.canvas.draw_idle()
            elif annotation.get_visible():
                annotation.set_visible(False)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)


def plot_lags(
    x: str,
    y: str,
    trials: List[FrozenTrial],
    gait_stats: Dict[str, np.ndarray],
    mean_speed: np.ndarray,
    args: argparse.Namespace,
):

    plt.figure(f"{x} vs {y} - {args.study_name}", figsize=args.figsize)
    plt.title(f"{x} vs {y}", fontsize=args.fontsize)

    plt.xlabel(f"{x}", fontsize=args.fontsize)
    plt.ylabel(f"{y}", fontsize=args.fontsize)

    scatter = plt.scatter(
        x=gait_stats[x],
        y=gait_stats[y],
        c=mean_speed,
        alpha=0.9,
        label="Trial",
        s=100,
        cmap=seaborn.color_palette("vlag", as_cmap=True),
    )

    create_annotation(scatter, trials, args)

    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    # plt.colorbar(label="Heading deviation (deg)")
    plt.colorbar(label="Mean speed (m/s)")

    plt.legend(loc="lower right", framealpha=0.9, fontsize=args.fontsize)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()


def plot_reached_angles(
    arrow_dx: np.ndarray,
    arrow_dy: np.ndarray,
    color_bar_values: np.ndarray,
    trials: List[FrozenTrial],
    args: argparse.Namespace,
):
    plt.figure(f"Reached angles - {args.study_name}", figsize=(9, 7))
    plt.title("Reached angles", fontsize=args.fontsize)

    # Unit circle
    plt.gca().add_patch(plt.Circle((0, 0), 1.0, color="black", fill=False))

    scatter = plt.scatter(
        x=arrow_dx,
        y=arrow_dy,
        c=color_bar_values,
        alpha=0.9,
        label="Trial",
        s=100,
        # cmap=seaborn.light_palette("seagreen", as_cmap=True),
    )

    create_annotation(scatter, trials, args)

    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.colorbar(label="Heading deviation (deg)")
    plt.legend(loc="lower right", framealpha=0.9, fontsize=args.fontsize)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()


def plot_reached_points(
    dx: np.ndarray,
    dy: np.ndarray,
    arrow_dx: np.ndarray,
    arrow_dy: np.ndarray,
    color_bar_values: np.ndarray,
    trials: List[FrozenTrial],
    args: argparse.Namespace,
):
    plt.figure(f"Reached points - {args.study_name}", figsize=args.figsize)
    plt.title("Reached points", fontsize=args.fontsize)

    plt.xlabel("dx", fontsize=args.fontsize)
    plt.ylabel("dy", fontsize=args.fontsize)

    plt.quiver(
        dx,
        dy,
        arrow_dx,
        arrow_dy,
        angles="uv",
        alpha=0.2,
    )

    scatter = plt.scatter(
        x=dx,
        y=dy,
        c=color_bar_values,
        alpha=0.9,
        label="Trial",
        s=100,
        cmap=seaborn.color_palette("vlag", as_cmap=True),
    )

    create_annotation(scatter, trials, args)

    plt.xlim(-1, 2)
    plt.ylim(-1, 1)
    # plt.colorbar(label="Heading deviation (deg)")
    plt.colorbar(label="Mean speed (m/s)")

    plt.legend(loc="lower right", framealpha=0.9, fontsize=args.fontsize)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()


def plot_umap(
    selected_keys: List[str],
    gait_stats: Dict[str, np.ndarray],
    color_bar_values: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    verbose: int = 1,
    args: Optional[argparse.Namespace] = None,
):
    assert args is not None
    print("Computing UMAP...")
    import umap

    data = np.concatenate([[gait_stats[key]] for key in selected_keys]).T

    umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, verbose=verbose)
    embedding = umap_model.fit_transform(data)

    name = f"n_neighbors={umap_model.n_neighbors} min_dist={umap_model.min_dist}"

    plt.figure(f"UMAP - {args.study_name}", figsize=args.figsize)
    plt.title(f"UMAP {name}", fontsize=args.fontsize)

    scatter = plt.scatter(
        x=embedding[:, 0],
        y=embedding[:, 1],
        c=color_bar_values,
        alpha=0.9,
        label="Trial",
        s=100,
        cmap=seaborn.color_palette("vlag", as_cmap=True),
    )

    create_annotation(scatter, trials, args)

    # plt.colorbar(label="Heading deviation (deg)")
    plt.colorbar(label="Mean speed (m/s)")
    plt.tight_layout()


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
    parser.add_argument("-umap", "--umap", help="Enable UMAP plot", action="store_true", default=False)
    parser.add_argument("-replay", "--sim-replay", help="Send trial to sim", action="store_true", default=False)
    parser.add_argument("--figsize", help="Figure size, width, height in inches.", nargs=2, type=int, default=[7.9, 5.4])
    parser.add_argument("--fontsize", help="Font size", type=int, default=16)
    args = parser.parse_args()

    # Enable LaTeX support
    if args.latex:
        plt.rc("text", usetex=True)

    trials = get_completed_trials(args)

    dx = get_values(trials, "mean_dx")
    dy = get_values(trials, "mean_dy")
    deviation = get_values(trials, "mean_deviation")
    mean_speed = get_values(trials, "mean_speed")
    n_steps = get_values(trials, "total_steps")

    # gait info
    # Pre-compute gait stats for all trials
    # transform list of dict to dict of numpy array
    gait_stats = gait_stats_to_dict([get_gait_stats(trial) for trial in trials])

    arrow_dx = np.cos(np.deg2rad(deviation))
    arrow_dy = np.sin(np.deg2rad(deviation))

    if args.umap:
        plot_umap(
            # selected_keys=["f_lag", "h_lag", "p_lag", "duty_factor"],
            selected_keys=["f_lag", "h_lag", "p_lag"],
            gait_stats=gait_stats,
            color_bar_values=mean_speed,
            n_neighbors=15,
            min_dist=0.1,
            verbose=1,
            args=args,
        )

    # Reach points in Cartesian space - with final orientation displayed
    plot_reached_points(
        dx,
        dy,
        arrow_dx,
        arrow_dy,
        color_bar_values=mean_speed,
        trials=trials,
        args=args,
    )

    # lag components
    plot_lags(
        x="f_lag",
        y="h_lag",
        trials=trials,
        gait_stats=gait_stats,
        mean_speed=mean_speed,
        args=args,
    )

    plot_reached_angles(
        arrow_dx,
        arrow_dy,
        color_bar_values=deviation,
        trials=trials,
        args=args,
    )

    plt.show()
