from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from gymnasium import spaces
import numpy as np
import simpy
from simulator.lora_simulator import LoraSimulator
import simulator.consts as consts
import simulator.utils as utils
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env

class LoRaEnvPZ(AECEnv):
    """
    Multi-agent LoRa Network Simulation for PettingZoo.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, nodes_count=10, data_size=16, avg_wake_up_time=30, sim_time=3600):
        super().__init__()

        self.nodes_count = nodes_count
        self.data_size = data_size
        self.avg_wake_up_time = avg_wake_up_time
        self.sim_time = sim_time

        self.simpy_env = simpy.Environment()
        self.simulator = LoraSimulator(nodes_count, data_size, avg_wake_up_time * 1000, sim_time * 1000, self.simpy_env)
        
        self.agents = [f"agent_{i}" for i in range(nodes_count)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, range(len(self.agents))))
        
        self.action_spaces = {agent: spaces.Discrete(3) for agent in self.agents}
        self.observation_spaces = {
            agent: spaces.Dict({
                "prr": spaces.Box(low=0, high=1, shape=(), dtype=np.float64),
                "rssi": spaces.Box(low=-200, high=0, shape=(), dtype=np.float64),
                "sf": spaces.Box(low=7, high=12, shape=(), dtype=np.int64)
            }) for agent in self.agents
        }

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

    def observe(self, agent):
        idx = self.agent_name_mapping[agent]
        return {
            "prr": consts.nodes[idx].prr_value,
            "rssi": consts.nodes[idx].rssi_value,
            "sf": consts.nodes[idx].sf_value
        }

    def step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)

        agent_index = self.agent_name_mapping[self.agent_selection]
        self.simulator.update_node_behavior(agent_index, action)
        self.simpy_env.run(until=self.current_step * 1000)

        # Update rewards and observations
        self.rewards[self.agent_selection] = self._calculate_reward(agent_index)
        self._accumulate_rewards()

        # Check if simulation is done
        self.current_step += 1
        if self.current_step >= self.sim_time:
            for agent in self.agents:
                self.dones[agent] = True

        self.agent_selection = self._agent_selector.next()

    def reset(self, **kwargs):
        self.simpy_env = simpy.Environment()
        self.simulator = LoraSimulator(self.nodes_count, self.data_size, self.avg_wake_up_time * 1000, self.sim_time * 1000, self.simpy_env)
        self.simulator.add_nodes()
        self.current_step = 0

        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def render(self, mode="human"):
        if mode == "human":
            print({agent: self.observe(agent) for agent in self.agents})

    def _calculate_reward(self, agent_index):
        lambda_value = 0.0001
        mean_prr = consts.nodes[agent_index].calculate_prr()
        retransmission_penalty = lambda_value * consts.nodes[agent_index].packets_sent_count
        return mean_prr - retransmission_penalty

# Wrap the PettingZoo environment for RLlib
def env_creator(env_config):
    return LoRaEnvPZ(**env_config)

# Register the environment with RLlib
register_env("lora_multi_agent", lambda config: PettingZooEnv(env_creator(config)))

# This registered environment can now be used in RLlib with the environment name "lora_multi_agent"
