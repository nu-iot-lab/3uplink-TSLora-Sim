import envs
# import gymnasium as gym
import gymnasium
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

env = gymnasium.make('lora-v0')

model = DQN('MlpPolicy', env, verbose=1)



