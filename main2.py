import gymnasium as gym
# import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

# Assuming you have an appropriate multi-agent version of your environment registered in Gym
class RewardLoggerCallback(BaseCallback):
    def __init__(self, check_freq):
        super(RewardLoggerCallback, self).__init__()
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

    env = gym.make("loraenv/LoRaMulti-v0", config={
        "nodes_count": nodes_count,
        "data_size": data_size,
        "avg_wake_up_time": avg_wake_up_time,
        "sim_time": sim_time,
    })

    models = {agent: DQN("MlpPolicy", env, verbose=1) for agent in env.possible_agents}

    # Train each model
    for agent, model in models.items():
        callback = RewardLoggerCallback(check_freq=100)
        model.learn(total_timesteps=int(sim_time * 100), callback=callback)
        model.save(f"{agent}_lora_model")
        plt.figure(figsize=(10, 5))
        plt.plot(callback.episode_rewards, marker="o", linestyle="-")
        plt.title(f"Total Reward per Episode During Training for {agent}")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(True)
        plt.savefig(f"{agent}_training_rewards.png")

    # Example evaluation phase for one agent
    test_env = env
    obs = test_env.reset()
    done = {agent: False for agent in test_env.possible_agents}
    while not all(done.values()):
        actions = {agent: models[agent].predict(obs[agent], deterministic=True)[0] for agent in test_env.possible_agents}
        obs, rewards, dones, _ = test_env.step(actions)
        done = dones

    # Close environment
    env.close()
