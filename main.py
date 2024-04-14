import sys
import gymnasium as gym
import loraenv
import simulator.utils as utils

from simulator.lora_simulator import LoraSimulator

from stable_baselines3 import DQN

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

        # Create a model or load a pre-trained model
        model = DQN("MultiInputPolicy", gym_env, verbose=1)
        # model = DQN.load("lora_model")

        # Training Phase
        # --------------
        utils.logging = False
        utils.log(f"!-- TRAINING START --!")
        # Calculate total timesteps for training
        episodes = 20
        total_timesteps = (
            sim_time * episodes
        )  # Assuming 1 timestep = 1 second in simulation
        model.learn(total_timesteps=total_timesteps, log_interval=4, progress_bar=True)
        model.save("lora_model")
        utils.log(f"!-- TRAINING END --!")

        # Evaluation Phase
        # ----------------
        utils.logging = True
        utils.log(f"!-- EVALUATION START --!")
        obs, info = gym_env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, terminated, info = gym_env.step(action)
            if done or terminated:
                utils.show_final_statistics()
                utils.log(f"!-- EVALUATION END --!")
                break

    else:
        print(
            "usage: ./main <number_of_nodes> <data_size(bytes)> <avg_wake_up_time(secs)> <sim_time(secs)>"
        )
        exit(-1)
