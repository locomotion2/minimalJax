import numpy as np

from Lagranx.src import lagranx as lx
import stable_baselines3.common.save_util as loader


class EnergyObserver:

    def __init__(self):
        params = loader.load_from_pkl(path='tmp/current', verbose=1)
        train_state = lx.create_train_state(0, 0, params=params)
        # self.lagrangian = lx.learned_lagrangian(params, train_state, output='lagrangian')
        self.energies = lx.partial(lx.learned_energies, params=params, train_state=train_state)
        self.kin_factors = np.array([1.785508155822754, 0.009507834911346436])
        self.pot_factors = np.array([1.785508155822754, -23.28985595703125])
    print('Using me!')

    def get_energies(self, q, dq):
        state = np.concatenate([q, dq])

        _, V, T, _, _ = self.energies(state)

        T = T * self.kin_factors[0] + self.kin_factors[1]
        V = V * self.pot_factors[0] + self.pot_factors[1]

        return T, V
