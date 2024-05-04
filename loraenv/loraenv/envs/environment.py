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
        avg_wake_up_time=30,
        sim_time=3600,
    ):
        super(LoRaEnv, self).__init__()

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
            self.avg_wake_up_time * 1000,
            self.sim_time * 1000,
            self.simpy_env,
        )

        # Action space: number of transmission slots (0 = 1, 1 = 2, 2 = 3)
        self.action_space = spaces.Discrete(3)

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

        # Other variables
        # self.previous_mean_prr = None
        # self.previous_packets_sent = None
        self.current_step = 0
        self.done = False
        self.truncated = False

    # Execute the action
    def step(self, action):
        if self.current_step == 0:
            self.simulator.start_simulation()
        if self.current_step >= self.sim_time:
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
        utils.log(
            f"!-- UPLINK NUMBER FOR STEP [{self.current_step}]: {action + 1} --!",
            self.simpy_env,
        )
        self.simulator.env.run(until=timestep)

        reward = self._calculate_reward(
            # self.previous_mean_prr, self.previous_packets_sent
        )
        obs = self._next_observation()
        info = {}

        # Check if the entire simulation duration has been reached
        self.done = self.current_step >= self.sim_time

        return obs, reward, self.done, self.truncated, info

    # Update the observation space
    def _next_observation(self):
        prr = np.array([node.prr_value for node in consts.nodes], dtype=np.float64)
        rssi = np.array([node.rssi_value for node in consts.nodes], dtype=np.float64)
        sf = np.array([node.sf_value for node in consts.nodes], dtype=np.int64)
        # print(
        #     f"Next Observation for STEP [{self.current_step}]:\nPRR: {prr}\nRSSI: {rssi}\nSF: {sf}\n"
        # )
        return {"prr": prr, "rssi": rssi, "sf": sf}

    # Reward formula
    def _calculate_reward(self):
        # Weight for penalizing retransmissions (0.001 = retransmissions over PRR, 0.0001 = PRR over retransmissions)
        lambda_value = 0.0005
        mean_prr = np.mean([node.calculate_prr() for node in consts.nodes])
        retransmission_penalty = lambda_value * sum(
            [node.packets_sent_count for node in consts.nodes]
        )
        reward = mean_prr - retransmission_penalty
        # print(f"Reward for STEP [{self.current_step}]: {reward:.3f}", self.simpy_env)
        return reward

    # def _calculate_reward(self, previous_mean_prr=None, previous_packets_sent=None):
    #     lambda_value = 0.0001  # Base value for the retransmission penalty
    #     prr_target = 0.98  # Target PRR that you find acceptable for the network

    #     # Calculate the current mean PRR
    #     mean_prr = np.mean([node.calculate_prr() for node in consts.nodes])

    #     # Calculate the retransmissions for the current step
    #     current_packets_sent = sum([node.packets_sent_count for node in consts.nodes])
    #     change_in_uplinks = current_packets_sent - (
    #         previous_packets_sent if previous_packets_sent is not None else 0
    #     )

    #     # Normalize the retransmissions
    #     uplink_penalty = lambda_value * change_in_uplinks

    #     # Dynamic adjustment of the lambda value based on performance
    #     lambda_value = (
    #         lambda_value / (prr_target / mean_prr)
    #         if mean_prr < prr_target
    #         else lambda_value * (mean_prr / prr_target)
    #     )

    #     # Calculate the change in PRR if we have a previous value to compare it with
    #     prr_improvement = mean_prr - (
    #         previous_mean_prr if previous_mean_prr is not None else mean_prr
    #     )

    #     # Combine the PRR improvement and the retransmission penalty
    #     reward = prr_improvement - uplink_penalty

    #     # Update the previous values
    #     self.previous_mean_prr = mean_prr
    #     self.previous_packets_sent = current_packets_sent

    #     # utils.log(
    #     #     f"Reward for STEP [{self.current_step}]: {reward:.3f}", self.simpy_env
    #     # )

    #     return reward

    # Reset the environment
    def reset(self, seed=None, options=None):
        self.simpy_env = simpy.Environment()
        self.simulator = LoraSimulator(
            self.nodes_count,
            self.data_size,
            self.avg_wake_up_time * 1000,
            self.sim_time * 1000,
            self.simpy_env,
        )
        # self.previous_mean_prr = None
        # self.previous_packets_sent = None
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
