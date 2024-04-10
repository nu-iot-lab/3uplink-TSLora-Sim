import gymnasium as gym
from gymnasium import spaces
import numpy as np
import simpy
from simulator.lora_simulator import LoraSimulator
import simulator.consts as consts
import simulator.utils as utils


class LoRaEnv(gym.Env):
    """
    Custom Environment for LoRa Network Simulation integrated with Reinforcement Learning.
    Follows the gym interface.
    """

    metadata = {"render_modes": ["console"]}

    def __init__(
        self,
        nodes_count=10,
        data_size=16,
        avg_wake_up_time=30 * 1000,
        sim_time=3600 * 1000,
    ):
        super(LoRaEnv, self).__init__()

        # Weight for penalizing retransmissions
        self.lambda_value = 0.0001

        # Actions: number of transmission slots (0 = 1, 1 = 2, 2 = 3)
        self.action_space = spaces.Discrete(3)

        # Setup simulation parameters
        self.nodes_count = nodes_count
        self.data_size = data_size
        self.avg_wake_up_time = avg_wake_up_time
        self.sim_time = sim_time

        # Setup simulator environment
        self.simpy_env = simpy.Environment()
        self.simulator = LoraSimulator(
            self.nodes_count,
            self.data_size,
            self.avg_wake_up_time,
            self.sim_time,
            self.simpy_env,
        )

        # Observation space: PRR, RSSI, SF for each node
        self.observation_space = spaces.Dict(
            {
                "prr": spaces.Box(
                    low=0, high=1, shape=(self.nodes_count,), dtype=np.float64
                ),
                "rssi": spaces.Box(
                    low=-200, high=0, shape=(self.nodes_count,), dtype=np.float64
                ),
                "sf": spaces.Box(
                    low=7, high=9, shape=(self.nodes_count,), dtype=np.int64
                ),
            }
        )
        self.current_step = 0
        self.done = False
        self.truncated = False

    # Update the observation space
    def _next_observation(self):
        prr = np.array([node.prr_value for node in consts.nodes], dtype=np.float64)
        rssi = np.array([node.rssi_value for node in consts.nodes], dtype=np.float64)
        sf = np.array([node.sf_value for node in consts.nodes], dtype=np.int64)
        # print(
        #     f"Next Observation for STEP [{self.current_step}]:\nPRR: {prr}\nRSSI: {rssi}\nSF: {sf}\n"
        # )
        return {"prr": prr, "rssi": rssi, "sf": sf}

    # Execute the action
    def step(self, action):
        if self.current_step == 0:
            self.simulator.start_simulation()
        if self.current_step >= (self.sim_time / 1000):
            self.done = True
            reward = self._calculate_reward()
            obs = self._next_observation()
            info = {}
            return obs, reward, self.done, info

        # Update number of transmissions
        self.simulator.update_nodes_behavior(action)

        # Advance the simulation by one second
        self.current_step += 1
        timestep = self.current_step * 1000
        print(f"!-- UPLINK NUMBER FOR STEP [{self.current_step}]: {action + 1} --!")
        self.simulator.env.run(until=timestep)

        reward = self._calculate_reward()
        obs = self._next_observation()
        info = {}

        # Check if the entire simulation duration has been reached
        self.done = self.current_step >= (self.sim_time / 1000)

        return obs, reward, self.done, self.truncated, info

    # Reward formula
    def _calculate_reward(self):
        mean_prr = np.mean([node.calculate_prr() for node in consts.nodes])
        retransmission_penalty = self.lambda_value * sum(
            [node.packets_sent_count for node in consts.nodes]
        )
        reward = mean_prr - retransmission_penalty
        return reward

    # Reset the environment
    def reset(self, seed=None, options=None):
        self.simpy_env = simpy.Environment()
        self.simulator = LoraSimulator(
            self.nodes_count,
            self.data_size,
            self.avg_wake_up_time,
            self.sim_time,
            self.simpy_env,
        )
        self.current_step = 0
        self.done = False
        self.truncated = False
        utils.reset_simulator()
        self.simulator.add_nodes()
        info = {}
        return self._next_observation(), info

    # Render the environment
    def render(self, mode="human"):
        print(self._next_observation())
