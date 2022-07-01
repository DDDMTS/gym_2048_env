from gym.envs.registration import register

register(
    id='2048game-v0',
    entry_point='gym_2048.envs:Env2048'
)