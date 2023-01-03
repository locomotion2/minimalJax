from gym.envs.registration import register

__version__ = "0.7.0"


register(
    id="WalkingBertSim-v1",
    entry_point="custom_envs.env.bert_walking_env:WalkingBertEnv",
    max_episode_steps=100,  # around 10s of interaction
    kwargs={
        "control_frequency": 10,  # 10Hz
        "threshold_center_deviation": 5,
        "weight_angular_velocity": 0.0,
        "limit_action_space_factor": 0.5,
        "weight_energy_consumption": 0.4,
    },
)

register(
    id="WalkingBertPolarSim-v1",
    entry_point="custom_envs.env.bert_walking_env:WalkingBertEnv",
    max_episode_steps=100,  # around 10s of interaction
    kwargs={
        "control_frequency": 10,  # 10Hz
        "threshold_center_deviation": 5,
        "weight_angular_velocity": 0.0,
        "weight_energy_consumption": 1.0,
        "use_polar_coords": True,
        "weight_distance_traveled": 50,
        "weight_linear_speed": 1,
    },
)

register(
    id="WalkingBertEnv-v1",
    entry_point="custom_envs.env.bert_walking_env:WalkingBertEnv",
    max_episode_steps=200,  # around 10s of interaction
    kwargs={"control_frequency": 20},  # 20Hz
)

register(
    id="TurningBertEnv-v1",
    entry_point="custom_envs.env.bert_turning_env:TurningBertEnv",
    max_episode_steps=200,  # around 10s of interaction
    kwargs={"control_frequency": 20},  # 20Hz
)

register(
    id="FixedTurningBertEnv-v2",
    entry_point="custom_envs.env.bert_fixed_turning_env:FixedTurningBertEnv",
    max_episode_steps=50,  # around 5s of interaction
    kwargs={
        "steps_at_target": 10,  # 1s at target
        "control_frequency": 10,
    },  # 10Hz
)

register(
    id="ModesBertSim-v1",
    entry_point="custom_envs.env.bert_modes:ModesBertEnv",
    max_episode_steps=200,  # around 5s of interaction
    kwargs={
        "control_frequency": 40,  # 40Hz
    },
)

register(
    id="ModesBertReal-v1",
    entry_point="custom_envs.env.bert_modes:ModesBertEnv",
    max_episode_steps=300,  # around 5s of interaction
    kwargs={
        "control_frequency": 60,  # capped at 60Hz because of the tracking system
        "is_real_robot": True,
    },
)

register(
    id="CPGBertReal-v1",
    entry_point="custom_envs.env.bert_cpg:CPGBertEnv",
    max_episode_steps=300,  # around 5s of interaction
    kwargs={
        "control_frequency": 60,  # capped at 60Hz because of the tracking system
        "is_real_robot": True,
    },
)


register(
    id="CPGWalkBertReal-v1",
    entry_point="custom_envs.env.bert_cpg:CPGBertEnv",
    max_episode_steps=300,  # around 5s of interaction
    kwargs={
        "control_frequency": 60,  # capped at 60Hz because of the tracking system
        "is_real_robot": True,
        "gait": "walk",
    },
)

register(
    id="CPGBertPronk-v1",
    entry_point="custom_envs.env.bert_cpg:CPGBertEnv",
    max_episode_steps=300,  # around 5s of interaction
    kwargs={
        "control_frequency": 60,  # capped at 60Hz because of the tracking system
        "is_real_robot": True,
        "gait": "pronk",
        "task": "balance",
    },
)

register(
    id="BertPronk-v1",
    entry_point="custom_envs.env.bert_cpg:CPGBertEnv",
    # Note: check max_episode_steps because of action repeat
    # real control freq is 30Hz but max episode steps is still 300
    max_episode_steps=300,  # around 5s of interaction
    kwargs={
        "control_frequency": 60,  # capped at 60Hz because of the tracking system
        "is_real_robot": True,
        "gait": "pronk",
        "enable_rl_offsets": True,
        "max_offset": 0.004,
        "task": "balance",
        "symmetry": None,
        "action_repeat": 2,
    },
)

register(
    id="BertPronkSpeed-v1",
    entry_point="custom_envs.env.bert_cpg:CPGBertEnv",
    max_episode_steps=300,  # around 5s of interaction
    kwargs={
        "control_frequency": 60,  # capped at 60Hz because of the tracking system
        "is_real_robot": True,
        "gait": "pronk",
        "enable_rl_offsets": True,
        "max_offset": 0.005,
        "task": "speed",
    },
)

register(
    id="BertTrotTurn-v1",
    entry_point="custom_envs.env.bert_cpg:CPGBertEnv",
    max_episode_steps=300,  # around 5s of interaction
    kwargs={
        "control_frequency": 60,  # capped at 60Hz because of the tracking system
        "is_real_robot": True,
        "gait": "trot",
        "enable_rl_offsets": True,
        "max_offset": 0.005,
        "task": "turn",
    },
)

register(
    id="BertTrotSpeed-v1",
    entry_point="custom_envs.env.bert_cpg:CPGBertEnv",
    max_episode_steps=300,  # around 5s of interaction
    kwargs={
        "control_frequency": 60,  # capped at 60Hz because of the tracking system
        "is_real_robot": True,
        "gait": "fast_trot",
        "enable_rl_offsets": True,
        "max_offset": 0.004,
        "task": "speed",
        "symmetry": "trot",
        "action_repeat": 2,
    },
)

register(
    id="ExploreCPGGait-v0",
    entry_point="custom_envs.env.bert_cpg:CPGBertEnv",
    max_episode_steps=300,  # around 5s of interaction
    kwargs={
        "control_frequency": 60,  # capped at 60Hz because of the tracking system
        "is_real_robot": True,
        "gait": "trot",
        "optimize_cpg_params": False,
        "explore_gaits": True,
    },
)

register(
    id="ExploreCPGGaitWithParams-v0",
    entry_point="custom_envs.env.bert_cpg:CPGBertEnv",
    max_episode_steps=300,  # around 5s of interaction
    kwargs={
        "control_frequency": 60,  # capped at 60Hz because of the tracking system
        "is_real_robot": True,
        "gait": "trot",
        "optimize_cpg_params": True,
        "explore_gaits": True,
        "optimize_only_omega": True,
    },
)
