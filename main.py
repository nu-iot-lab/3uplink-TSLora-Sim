import sys
import gymnasium as gym
import loraenv

from simulator.lora_simulator import LoraSimulator
from simulator.utils import show_final_statistics

from stable_baselines3 import DQN

# Gymnasium environment
gym_env = gym.make("loraenv/LoRa-v0")

# DQN model
# model = DQN("MultiInputPolicy", gym_env, verbose=1)
# model.learn(total_timesteps=10, log_interval=4)
# model.save("lora_model")

if __name__ == "__main__":
    if len(sys.argv) == 5:
        nodes_count = int(sys.argv[1])
        data_size = int(sys.argv[2])
        avg_wake_up_time = int(sys.argv[3]) * 1000
        sim_time = int(sys.argv[4]) * 1000

        gym_env.setup(
            nodes_count=nodes_count,
            data_size=data_size,
            avg_wake_up_time=avg_wake_up_time,
            sim_time=sim_time,
        )

        # Training Phase
        # --------------
        print("!-- TRAINING START --!")
        model = DQN("MultiInputPolicy", gym_env, verbose=1)

        # Calculate total timesteps for training
        episodes = 10
        total_timesteps = (
            sim_time / 1000 * episodes
        )  # Assuming 1 timestep = 1 second in simulation
        model.learn(total_timesteps=int(total_timesteps), log_interval=4)
        model.save("lora_model")
        print("!-- TRAINING END --!")

        # Evaluation Phase
        # ----------------
        print("!-- EVALUATION START --!")
        # Load the trained model (optional if you just trained it)
        # model = DQN.load("lora_model")
        obs, info = gym_env.reset()

        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, terminated, info = gym_env.step(action)
            if done or terminated:
                show_final_statistics()
                obs, info = gym_env.reset()
                break

        print("!-- EVALUATION END --!")

    else:
        print(
            "usage: ./main <number_of_nodes> <data_size(bytes)> <avg_wake_up_time(secs)> <sim_time(secs)>"
        )
        exit(-1)
