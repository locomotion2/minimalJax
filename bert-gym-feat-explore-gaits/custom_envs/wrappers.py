import gym
import links_and_nodes as ln
import numpy as np


class MetricWrapper(gym.Wrapper):
    """
    Wrapper to compute different metrics:
    - mean speed
    - mean energy cost (CoT with electrical cost)
    - mean distance travelled

    :param env: Gym env to wrap.
    :param min_steps: Wait for a minimum of steps before recording energy/distance
        mainly because the computation will be innacurate at the beginning (low speed)
    :param verbose:
    """

    def __init__(self, env: gym.Env, min_steps: int = 15, verbose: int = 0):
        super().__init__(env)
        self.energy_cost_history = []
        self.mecha_cost_history = []
        self.mean_speed_history = []
        self.distance_travelled_history = []
        self.dy_history = []
        self.heading_deviation_history = []

        # Connect to ln manager
        self.ln_client = ln.client("metric_wrapper_client")
        self.distance_service = self.ln_client.get_service("bert.distance_reward_service", "distance_reward_service")
        self.energy_service = self.ln_client.get_service("bert.energy_cost_service", "energy_cost_service")
        self.verbose = verbose
        self.min_steps = min_steps
        self.n_steps = 0

    def reset_metrics(self):
        self.energy_cost_history = []
        self.mecha_cost_history = []
        self.mean_speed_history = []
        self.distance_travelled_history = []
        self.dy_history = []
        self.heading_deviation_history = []

    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        self.n_steps = 0
        return obs

    def step(self, action):
        self.n_steps += 1
        if self.n_steps == self.min_steps:
            # Toggle recording
            self.distance_service.call()
            self.energy_service.call()

        obs, reward, done, info = self.env.step(action)
        if done:
            # Toggle recording
            self.distance_service.call()
            self.energy_service.call()
            self.distance_travelled_history.append(self.distance_service.resp.dx)
            self.dy_history.append(self.distance_service.resp.dy)
            self.heading_deviation_history.append(self.distance_service.resp.heading_deviation)
            self.mean_speed_history.append(self.distance_service.resp.mean_speed)
            self.energy_cost_history.append(self.energy_service.resp.mean_cost_elec)
            self.mecha_cost_history.append(self.energy_service.resp.mean_cost_mecha)

        return obs, reward, done, info

    @property
    def mean_energy_cost(self) -> float:
        return np.mean(self.energy_cost_history)

    @property
    def mean_speed(self) -> float:
        return np.mean(self.mean_speed_history)

    @property
    def mean_distance_travelled(self) -> float:
        """in meters (m)."""
        return np.mean(self.distance_travelled_history)

    @property
    def mean_dy(self) -> float:
        return np.mean(self.dy_history)

    @property
    def mean_heading_deviation(self) -> float:
        """in degrees (deg)."""
        return np.mean(self.heading_deviation_history)

    def print_metrics(self):
        mean_dx = self.mean_distance_travelled
        std_dx = np.std(self.distance_travelled_history)
        std_speed = np.std(self.mean_speed_history)
        std_dy = np.std(self.dy_history)
        std_heading = np.std(self.heading_deviation_history)
        print(f"dx= {mean_dx * 100:.2f} +/- {std_dx:.2f} cm | mean_speed= {self.mean_speed:.2f} +/- {std_speed:.2f} m/s")
        print(
            f"dy= {self.mean_dy * 100:.2f} +/- {std_dy:.2f} cm | "
            f"mean_heading_deviation= {self.mean_heading_deviation:.2f} +/- {std_heading:.2f} m/s"
        )

        std_energy = np.std(self.energy_cost_history)
        mean_mecha = np.mean(self.mecha_cost_history)
        std_mecha = np.std(self.mecha_cost_history)
        print(f"Mean cost elec: {self.mean_energy_cost:.4f} +/- {std_energy:.4f}")
        print(f"Mean cost mecha = {mean_mecha:.4f} +/- {std_mecha:.4f} ")
