import gym
from gym_2048.envs.env2048 import Env2048

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
oenv = make_vec_env(Env2048)
env = VecFrameStack(oenv, n_stack=2)

model = DQN.load("model\dqn_2048.zip")
max_block = dict()
obs = env.reset()

for i in range(100):
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            temp = info[0]['max_block']
            if temp not in max_block.keys():
                max_block[temp] = 1
            else:
                max_block[temp] += 1
            obs = env.reset()
            break

print(max_block)