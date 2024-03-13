import sys
import gymnasium as gym 
import loraenv

from simulator.lora_simulator import LoraSimulator
from simulator.singleton import EnvironmentSingleton

from simulator.utils import show_final_statistics

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN


# simpy environment
env = EnvironmentSingleton.get_instance()

# gym environment
gym_env = gym.make('loraenv/LoRa-v0') 

if __name__ == '__main__':
    if len(sys.argv) == 5:
        nodes_count = int(sys.argv[1])
        data_size = int(sys.argv[2])
        avg_wake_up_time = int(sys.argv[3])
        sim_time = int(sys.argv[4])


        avg_wake_up_time *= 1000
        sim_time *= 1000

        simulator = LoraSimulator(nodes_count=nodes_count, data_size=data_size, avg_wake_up_time=avg_wake_up_time,
                              sim_time=sim_time)

        simulator.add_nodes()

        simulator.start_simulation()

        show_final_statistics()
        


    else:
        print("usage: ./main <number_of_nodes> <data_size(bytes)> <avg_wake_up_time(secs)> <sim_time(secs)>")
        exit(-1)
