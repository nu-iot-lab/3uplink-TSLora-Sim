import gymnasium as gym
from gymnasium import spaces
import numpy as np
import simpy
from simulator.consts import nodes
from simulator.lora_simulator import LoraSimulator


class LoRaEnv(gym.Env):
    """
    Custom Environment for LoRa Network Simulation integrated with Reinforcement Learning.
    Follows the gym interface.
    """

    metadata = {"render_modes": ["console"]}

    def __init__(self):
        super(LoRaEnv, self).__init__()

        self.lambda_value = 0.01  # Weight for penalizing retransmissions

        # Actions: number of transmission slots (1, 2, 3)
        self.action_space = spaces.Discrete(3)

        self.nodes_count = 10
        self.data_size = 16
        self.avg_wake_up_time = 30
        self.sim_time = 3600

        self.observation_space = spaces.Dict(
            {
                "prr": spaces.Box(
                    low=0, high=1, shape=(self.nodes_count,), dtype=np.float64
                ),
                "rssi": spaces.Box(
                    low=-200, high=0, shape=(self.nodes_count,), dtype=np.int64
                ),
                "sf": spaces.Box(
                    low=7, high=12, shape=(self.nodes_count,), dtype=np.int64
                ),
            }
        )

        self.simpy_env = None
        self.simulator = None
        self.current_step = 0
        self.done = False
        self.truncated = False

    def _next_observation(self):
        prr = np.array([node.calculate_prr() for node in nodes])
        rssi = np.array([node.rssi_value for node in nodes])
        sf = np.array([node.sf for node in nodes])
        print(f"Next Observation:\nPRR: {prr}\nRSSI: {rssi}\nSF: {sf}\n")
        return {"prr": prr, "rssi": rssi, "sf": sf}

    def step(self, action):
        if self.current_step >= self.sim_time:
            self.done = True
            reward = self._calculate_reward()
            obs = self._next_observation()
            return obs, reward, self.done, {}

        self.simulator.update_nodes_behavior(action)

        # Advance the simulation by one second
        self.current_step += 1

        # self.simulator.start_simulation(self.current_step)

        reward = self._calculate_reward()
        obs = self._next_observation()
        info = {}
        # Check if the entire simulation duration has been reached
        self.done = self.current_step >= self.sim_time

        return obs, reward, self.done, self.truncated, info

    def _calculate_reward(self):
        return 1

    def reset(self, seed=None, options=None):
        self.simpy_env = simpy.Environment()
        self.current_step = 0
        self.done = False
        self.simulator = LoraSimulator(
            self.nodes_count,
            self.data_size,
            self.avg_wake_up_time,
            self.sim_time,
            self.simpy_env,
        )
        self.simulator.add_nodes()
        self.simulator.start_simulation()
        info = {}
        return self._next_observation(), info

    def render(self, mode="human"):
        print(self._next_observation())


# Example of creating and testing the environment with a specific penalty coefficient
# env = LoRaEnv(sf=7, )
# env = LoRaEnv(num_agents=10, data_size=100, avg_wake_up_time=5, sim_time=100)
# initial_state = env.reset()
# print(f"Initial State: RSSI={initial_state[0]:.2f} dBm, PRR={initial_state[1]:.2f}")

# env = LoRaEnv()
# for _ in range(10):
#     obs = env.reset()
#     while True:
#         action=np.array([7])
#         obs, r, done, _ = env.step(action, 0)
#         print(obs)
#         if done:
#             break
