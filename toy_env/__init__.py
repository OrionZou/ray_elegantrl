from gym.envs.registration import register

register(
    id='OneDimGoalEnv-v0',
    entry_point='safety_env.safety_env:OneDimGoalEnv',
)