__all__ = ["models", "controllers", "learning", "environments"]

from gymnasium.envs.registration import register

register(
    id="TestEnvironment-v1",
    entry_point="discovery.gym_environment:BaseGymEnvironment",
    kwargs={
        "render": False,
        "energy_command": None,
        "mode": 'random_des',
        "solve": False,
    },
)