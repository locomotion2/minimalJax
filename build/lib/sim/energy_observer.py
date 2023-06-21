import numpy as np

from src import lagranx as lx
import stable_baselines3.common.save_util as loader


class EnergyObserver:

    def __init__(self):
        params = loader.load_from_pkl(path='tmp/current', verbose=1)
        train_state = lx.create_train_state(0, 0, params=params)
        self.lagrangian = lx.energy_func(params, train_state, output='lagrangian')
        self.energies = lx.energy_func(params, train_state, output='energies')
        self.kin_factors = np.array([3.5177876949310303, -0.012919038534164429])
        self.pot_factors = np.array([0.9047878384590149, 0.540073573589325])

    def get_energies(self, q, dq):
        state = np.concatenate([q, dq])

        T = lx.kin_energy_lagrangian(state, lagrangian=self.lagrangian)
        _, V = self.energies(q, dq)

        T = T * self.kin_factors[0] + self.kin_factors[1]
        V = V * self.pot_factors[0] + self.pot_factors[1]

        return T, V
