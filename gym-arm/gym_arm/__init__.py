from gym.envs.registration import register

register(
    id='Arm-v0',
    entry_point='gym_arm.envs:ArmEnv',
    max_episode_steps=50,
)
