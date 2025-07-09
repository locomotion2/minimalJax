# src/learning/curricula.py

import jax
import jax.numpy as jnp
from src.CONSTANTS import *

class Curriculum:
    def __init__(self,
                 max_difficulty: float = 1.0,
                 min_difficulty: float = 0.0,
                 ):
        self.max_difficulty = max_difficulty
        self.min_difficulty = min_difficulty

    def get_difficulty(self):
        raise NotImplementedError


class UniformGrowthCurriculum(Curriculum):
    def __init__(self,
                 initial_difficulty: float = 0.0,
                 max_difficulty: float = 1.0,
                 min_difficulty: float = 0.0,
                 ):
        super().__init__(max_difficulty, min_difficulty)
        self.difficulty = initial_difficulty
        # Fix: Use jax.random for PRNG key initialization
        self.rng = jax.random.PRNGKey(0) # You can use a more dynamic seed if needed

    def get_difficulty(self):
        return self.difficulty

    def update(self, success_rate):
        self.difficulty = jnp.clip(self.difficulty + success_rate / 100, # Use jnp.clip
                                   self.min_difficulty,
                                   self.max_difficulty)


class ThresholdGrowthCurriculum(Curriculum):
    def __init__(self,
                 threshold: float = 0.8,
                 growth_factor: float = 1.05,
                 initial_difficulty: float = 0.0,
                 max_difficulty: float = 1.0,
                 min_difficulty: float = 0.0,
                 ):
        super().__init__(max_difficulty, min_difficulty)
        self.difficulty = initial_difficulty
        self.threshold = threshold
        self.growth_factor = growth_factor
        # Fix: Use jax.random for PRNG key initialization
        self.rng = jax.random.PRNGKey(0) # You can use a more dynamic seed if needed

    def get_difficulty(self):
        return self.difficulty

    def update(self, success_rate):
        if success_rate > self.threshold:
            self.difficulty = jnp.clip(self.difficulty * self.growth_factor, # Use jnp.clip
                                       self.min_difficulty,
                                       self.max_difficulty)


class RandomCurriculum(Curriculum):
    def __init__(self,
                 initial_difficulty: float = 0.0,
                 max_difficulty: float = 1.0,
                 min_difficulty: float = 0.0,
                 ):
        super().__init__(max_difficulty, min_difficulty)
        self.difficulty = initial_difficulty
        # Fix: Use jax.random for PRNG key initialization
        self.rng = jax.random.PRNGKey(0) # You can use a more dynamic seed if needed

    def get_difficulty(self):
        # Fix: Use jax.random.uniform and split the key
        self.rng, subkey = jax.random.split(self.rng)
        return jax.random.uniform(subkey, minval=self.min_difficulty, maxval=self.max_difficulty)

    def update(self, success_rate):
        pass


class NoneCurriculum(Curriculum):
    def __init__(self,
                 difficulty: float = 0.0,
                 max_difficulty: float = 1.0,
                 min_difficulty: float = 0.0,
                 ):
        super().__init__(max_difficulty, min_difficulty)
        self.difficulty = difficulty

    def get_difficulty(self):
        return self.difficulty

    def update(self, success_rate):
        pass