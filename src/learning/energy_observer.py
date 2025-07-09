import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

from identification.src.training import trainer
from Identification.src.dynamix import wrappings
import stable_baselines3.common.save_util as loader


class EnergyObserver:

    def __init__(self, settings):
        # TODO: Assume that sys_utils is given through the settings
        params = loader.load_from_pkl(path='tmp/current', verbose=1)
        train_state = trainer.create_train_state(settings, 0, params=params)
        self.energies = wrappings.build_energy_call(settings,
                                                    params,
                                                    train_state)
    print('Using the trained energy observer.')

    # TODO: all this will be deleted
    def get_energies(self, q, dq):
        state = jnp.concatenate([q, dq])

        _, V, T, _, _ = self.energies(state)

        return T, V
