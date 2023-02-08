from gym.envs.registration import register

register(
    id="TestEnvironment-v1",
    entry_point="sim.gym_environment:BaseGymEnvironment",
    kwargs={
    },
)