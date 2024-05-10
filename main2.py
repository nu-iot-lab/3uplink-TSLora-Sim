import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

from pettingzoo.test import parallel_api_test
from multienv.multienv_v0 import LoRaEnvParallel
from pettingzoo.utils import parallel_to_aec
from pettingzoo.utils.wrappers import BaseWrapper
from stable_baselines3.common.env_util import make_vec_env
from supersuit import pad_action_space_v0, pad_observations_v0
import supersuit as ss

class RewardLoggerCallback(BaseCallback):
    def __init__(self, check_freq, verbose=1):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            rewards = self.training_env.get_attr('rewards')
            self.episode_rewards.append(rewards)
        return True

if __name__ == "__main__":
    nodes_count = 10  # Example parameter
    data_size = 16    # Example parameter
    avg_wake_up_time = 30  # Example parameter
    sim_time = 3600   # Example parameter

    env = LoRaEnvParallel(nodes_count=nodes_count, data_size=data_size, avg_wake_up_time=avg_wake_up_time, sim_time=sim_time)
    parallel_api_test(env, num_cycles=1000)
    # aec_env = env
    # # wrapped_env = pad_action_space(pad_observations(aec_env))

    # # vec_env = pettingzoo_env_to_vec_env_v1(wrapped_env)
    # wrapped_env = pad_observations_v0(pad_action_space_v0(aec_env))

    # vec_env = ss.pettingzoo_env_to_vec_env_v1(wrapped_env)
    # # aec_env = ss.concat_vec_envs_v1(aec_env, 8, num_cpus=1, base_class="stable_baselines3")

    # model = DQN("MultiInputPolicy", vec_env, verbose=1)
    # reward_logger = RewardLoggerCallback(check_freq=100)

    # # Training Phase
    # model.learn(total_timesteps=100, callback=reward_logger)
    # model.save("dqn_lora_model")

    # # Plot the rewards collected during the training
    # plt.figure(figsize=(10, 5))
    # plt.plot(reward_logger.episode_rewards, marker="o", linestyle="-")
    # plt.title("Total Reward per Episode During Training")
    # plt.xlabel("Episode")
    # plt.ylabel("Total Reward")
    # plt.grid(True)
    # plt.savefig("training_rewards.png")

    # # Evaluation Phase
    # obs = vec_env.reset()
    # done = False
    # while not done:
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = vec_env.step(action)
    #     vec_env.render()
