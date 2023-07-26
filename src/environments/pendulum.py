from src.environments.base_environments import CPGEnv, DirectEnv
from src.models.dpendulum import Pendulum


class PendulumCPGEnv(CPGEnv):
    def __init__(self, params: dict = None):
        super().__init__(params=params)

        # Build components
        self.model = Pendulum(params=self.config.get('model_params'))


class PendulumDirectEnv(DirectEnv):
    def __init__(self, params: dict = None):
        super().__init__(params=params)

        # Build components
        self.model = Pendulum(params=self.config.get('model_params'))
