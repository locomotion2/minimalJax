from src.environments.base_environments import CPGEnv, DirectEnv
from src.models.dpendulum import DoublePendulum


class DoublePendulumCPGEnv(CPGEnv):
    def __init__(self, params: dict = None):
        super().__init__(params=params)

        # Build components
        self.model = DoublePendulum(params=self.config.get('model_params'))


class DoublePendulumDirectEnv(DirectEnv):
    def __init__(self, params: dict = None):
        super().__init__(params=params)

        # Build components
        self.model = DoublePendulum(params=self.config.get('model_params'))
