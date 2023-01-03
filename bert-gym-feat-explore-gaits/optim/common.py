import warnings

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)


class RandomAgent(object):
    def __init__(self, env):
        super(RandomAgent, self).__init__()
        self.env = env

    def predict(self, obs, *args, **kwargs):
        return self.env.action_space.sample(), None


class PlotFootTrajectoryCallback(object):
    def __init__(self, env, leg_idx=0):
        super().__init__()
        self.desired_radius = []
        self.desired_angles = []
        self.radius = []
        self.angles = []
        self.env = env
        self.leg_idx = leg_idx

    def __call__(self, _locals, _globals):
        self.desired_radius.append(self.env.current_radii[self.leg_idx])
        self.desired_angles.append(self.env.current_angles[self.leg_idx])
        self.radius.append(self.env.measured_radii[self.leg_idx])
        self.angles.append(self.env.measured_angles[self.leg_idx])

    def plot(self):
        self.radius = np.array(self.radius)
        self.angles = np.array(self.angles)
        self.desired_radius = np.array(self.desired_radius)
        self.desired_angles = np.array(self.desired_angles)
        plt.figure("Foot Trajectory")
        plt.xlabel("x (m)")
        plt.ylabel("z (m)")
        plt.plot(self.radius * np.sin(self.angles), -self.radius * np.cos(self.angles), label="measured")
        plt.plot(
            self.desired_radius * np.sin(self.desired_angles),
            -self.desired_radius * np.cos(self.desired_angles),
            label="desired",
        )
        plt.legend()

        fig, axes = plt.subplots(2)
        fig.suptitle("Radius and angle")
        axes[0].plot(np.arange(len(self.radius)), self.radius, label="measured")
        axes[0].plot(np.arange(len(self.desired_radius)), self.desired_radius, label="desired")
        axes[1].plot(np.arange(len(self.angles)), np.rad2deg(self.angles), label="measured")
        axes[1].plot(np.arange(len(self.desired_angles)), np.rad2deg(self.desired_angles), label="desired")
        plt.legend()
        plt.show()
