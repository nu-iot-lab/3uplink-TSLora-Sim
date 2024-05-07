import sys
import gymnasium as gym
import loraenv
import simulator.utils as utils
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
        rewards_per_evaluation = []
        # mean_prrs_rl = []
        # uplink_count_rl = []
        total_reward = 0
        done = False
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, terminated, info = gym_env.step(action)
            total_reward += reward
            rewards_per_evaluation.append(reward)  # Log each reward
            # mean_prr = np.mean([node.calculate_prr() for node in consts.nodes])
            # mean_prrs_rl.append(mean_prr)
            # uplink_count = sum([node.packets_sent_count for node in consts.nodes])
            # uplink_count_rl.append(uplink_count)
            if done or terminated:
                utils.show_final_statistics()
                utils.log(f"!-- EVALUATION END --!")
                break

        # utils.logging = True
        # utils.log(f"!-- EVALUATION START --!")
        # obs, info = gym_env.reset()
        # mean_prrs_default = []
        # uplink_count_default = []
        # done = False
        # while True:
        #     action = 2
        #     obs, reward, done, terminated, info = gym_env.step(action)
        #     mean_prr = np.mean([node.calculate_prr() for node in consts.nodes])
        #     mean_prrs_default.append(mean_prr)
        #     uplink_count = sum([node.packets_sent_count for node in consts.nodes])
        #     uplink_count_default.append(uplink_count)
        #     if done or terminated:
        #         utils.show_final_statistics()
        #         utils.log(f"!-- EVALUATION END --!")
        #         break

        # # Plot the mean PRR comparison
        # plt.figure(figsize=(10, 5))
        # plt.plot(mean_prrs_default, label="Triple Uplink", marker="o", linestyle="-")
        # plt.plot(mean_prrs_rl, label="Dynamic Uplink", marker="x", linestyle="--")
        # plt.title("Mean PRR vs Time")
        # plt.xlabel("Time")
        # plt.ylabel("Mean PRR")
        # plt.legend()
        # plt.grid(True)
        # plt.savefig("mean_prrs.png")

        # # Plot the total uplink count comparison
        # plt.figure(figsize=(10, 5))
        # plt.plot(uplink_count_default, label="Triple Uplink", marker="o", linestyle="-")
        # plt.plot(uplink_count_rl, label="Dynamic Uplink", marker="x", linestyle="--")
        # plt.title("Total Uplink Count vs Time")
        # plt.xlabel("Time")
        # plt.ylabel("Total Uplink Count")
        # plt.legend()
        # plt.grid(True)
        # plt.savefig("total_uplink_count.png")

        # Plot the rewards collected during the evaluation
        plt.figure(figsize=(10, 5))
        plt.plot(
            range(1, len(rewards_per_evaluation) + 1),
            rewards_per_evaluation,
            marker="o",
            linestyle="-",
        )
        plt.title("Rewards per Step During Evaluation")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.savefig("evaluation_phase.png")

    else:
        print(
            "usage: ./main <number_of_nodes> <data_size(bytes)> <avg_wake_up_time(secs)> <sim_time(secs)>"
        )
        exit(-1)
