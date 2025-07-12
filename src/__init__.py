__all__ = ["models", "controllers", "learning", "environments"]

from gymnasium.envs.registration import register

register(
    id="TestEnvironment-v1",
    entry_point="src.learning.gym_environment:BaseGymEnvironment",
    kwargs={
        "render": False,
        "energy_command": 0.0,
        "mode": 'random_des',
        "solve": False,
    },
)