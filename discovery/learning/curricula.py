from discovery.CONSTANTS import *

import numpy as np

from abc import ABC, abstractmethod


class BaseCurriculum(ABC):

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def grow(self):
        pass

    @abstractmethod
    def check_criterion(self, score, E):
        pass


class UniformGrowthCurriculum(BaseCurriculum):
    def __init__(self, starting_range: list = None, growth_rate: float = GROWTH_RATE, min_score: float = MIN_SCORE):
        if starting_range is None:
            self.range = [0, 0]
        else:
            self.range = starting_range
        self.growth_rate = growth_rate
        self.min_score = min_score

        self.rng = np.random.default_rng()
        self.E_max = self.range[1]

    def sample(self):
        return self.rng.uniform(self.range[0], self.range[1])

    def grow(self):
        if self.range[1] < MAX_GROWTH:
            self.range[1] += self.growth_rate
            print(f"New max. energy: {self.range[1]: .2f}J")

    def check_criterion(self, score, E):
        if score >= self.min_score and self.E_max - self.growth_rate < E:
            if self.E_max < E:
                self.E_max = E
            return True
        else:
            return False
