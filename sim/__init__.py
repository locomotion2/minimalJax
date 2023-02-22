from gym.envs.registration import register

register(
    id="TestEnvironment-v1",
    entry_point="sim.gym_environment:BaseGymEnvironment",
    kwargs={
        "render": False,
        "energy_command": None,
        "mode": 'equilibrium',
        "solve": False,
    },
)