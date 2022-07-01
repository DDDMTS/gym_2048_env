# gym_2048

This is the Gym environment for the 2048 game.

## how to use

I strongly recommend you use this environment by passing keyword arguments to Gym, such as:
~~~ python
import gym
import gym_2048.envs
env = gym.make("2048game-v0")
~~~
Rather than install it, because I didn't test this part :)

## test result 

I trained the A2C algorithm using this algorithm for 500k rounds, the max block I got was 128. 

## future work

- Continue to test this environment using more algorithms and rounds.
- Finish the rendering part for this environment.
- Try to use another way to give rewards and find whether it can speed the learning speed.

