import sys
import gymnasium as gym
import loraenv
import simulator.utils as utils
import simulator.consts as consts
import matplotlib.pyplot as plt

import numpy as np
import simulator.consts as consts

from simulator.lora_simulator import LoraSimulator
from reward_caller_callback import RewardLoggerCallback

from stable_baselines3 import PPO

if __name__ == "__main__":
    if len(sys.argv) == 5:
        nodes_count = int(sys.argv[1])
        data_size = int(sys.argv[2])
        avg_wake_up_time = int(sys.argv[3])
        sim_time = int(sys.argv[4])

        consts.nodes_count = nodes_count
        consts.data_size = data_size
        consts.avg_wake_up_time = avg_wake_up_time
        consts.sim_time = sim_time

        # Gymnasium environment
        gym_env = gym.make(
            "loraenv/LoRa-v0",
            nodes_count=nodes_count,
            data_size=data_size,
            avg_wake_up_time=avg_wake_up_time,
            sim_time=sim_time,
        )

        train = False
        if train:
            # Create new model
            model = PPO("MultiInputPolicy", gym_env, verbose=1)
            reward_logger = RewardLoggerCallback()

            # Training Phase
            # --------------
            utils.logging = False
            utils.log(f"!-- TRAINING START --!")
            # Calculate total timesteps for training
            episodes = 10
            total_timesteps = (
                sim_time * episodes
            )  # Assuming 1 timestep = 1 second in simulation
            model.learn(
                total_timesteps=total_timesteps,
                log_interval=4,
                progress_bar=True,
                callback=reward_logger,
            )
            model.save("lora_model")
            utils.log(f"!-- TRAINING END --!")

            # Plot the rewards collected during the training
            plt.figure(figsize=(10, 5))
            plt.plot(reward_logger.episode_rewards, marker="o", linestyle="-")
            plt.title("Total Reward per Episode During Training")
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.grid(True)
            plt.savefig("training_phase.png")

        # Evaluation Phase
        # ----------------
        model = PPO.load("lora_model")
        utils.log(f"!-- EVALUATION START --!")
        obs, info = gym_env.reset()
        rewards_per_evaluation = [
            [] for _ in range(nodes_count)
        ]  # List to hold rewards for each node
        total_rewards_per_node = [
            0
        ] * nodes_count  # List to hold total rewards for each node

        done = False
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, terminated, info = gym_env.step(action)
            for i in range(nodes_count):
                rewards_per_evaluation[i].append(
                    reward[i]
                )  # Log each reward for each node
                total_rewards_per_node[i] += reward[i]  # Sum rewards for each node

            if done or terminated:
                utils.show_final_statistics()
                utils.log(f"!-- EVALUATION END --!")
                break

        # Plot the rewards collected during the evaluation for each node
        plt.figure(figsize=(10, 5))
        for i in range(nodes_count):
            plt.plot(
                range(1, len(rewards_per_evaluation[i]) + 1),
                rewards_per_evaluation[i],
                marker="o",
                linestyle="-",
                label=f"Node {i+1}",
            )
        plt.title("Rewards per Step During Evaluation for Each Node")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        plt.savefig("evaluation_phase_per_node.png")

    else:
        print(
            "usage: ./main <number_of_nodes> <data_size(bytes)> <avg_wake_up_time(secs)> <sim_time(secs)>"
        )
        exit(-1)
