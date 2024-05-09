import simpy
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec
from simulator.lora_simulator import LoraSimulator
import simulator.consts as consts

class LoRaEnvParallel(ParallelEnv):
    metadata = {'render_modes': ['human'], 'name': 'lora_v1'}

    def __init__(self, nodes_count=10, data_size=16, avg_wake_up_time=30, sim_time=3600, render_mode=None):
        self.possible_agents = [f"agent_{i}" for i in range(nodes_count)]
        self.agent_name_mapping = dict(zip(self.possible_agents, range(len(self.possible_agents))))
        self.render_mode = render_mode
        self.nodes_count = nodes_count
        self.data_size = data_size
        self.avg_wake_up_time = avg_wake_up_time
        self.sim_time = sim_time

    def observation_space(self, agent):
        return spaces.Dict({
            "prr": spaces.Box(low=0, high=1, shape=(), dtype=np.float32),
            "rssi": spaces.Box(low=-200, high=0, shape=(), dtype=np.float32),
            "sf": spaces.Box(low=7, high=12, shape=(), dtype=np.int64)
        })

    def action_space(self, agent):
        return spaces.Discrete(3)

    def reset(self, **kwargs):
        self.simpy_env = simpy.Environment()
        self.simulator = LoraSimulator(self.nodes_count, self.data_size, self.avg_wake_up_time * 1000, self.sim_time * 1000, self.simpy_env)
        self.simulator.add_nodes()
        self.agents = self.possible_agents[:]
        self.current_step = 0
        self.dones = {agent: False for agent in self.agents}  # Initialize dones for all agents
        observations = {agent: self.observe(agent) for agent in self.agents}
        return observations


    def step(self, actions):
        for agent in actions:
            agent_index = self.agent_name_mapping[agent]
            if not self.dones[agent]:
                self.simulator.update_node_behavior(agent_index, actions[agent])

        self.simpy_env.run(until=self.current_step * 1000)
        self.current_step += 1

        observations = {agent: self.observe(agent) for agent in self.agents}
        rewards = {agent: self._calculate_reward(self.agent_name_mapping[agent]) for agent in self.agents}
        dones = {agent: self.current_step >= self.sim_time for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        if self.render_mode == 'human':
            self.render()

        return observations, rewards, dones, dones, infos

    def observe(self, agent):
        idx = self.agent_name_mapping[agent]
        return {
            "prr": consts.nodes[idx].prr_value,
            "rssi": consts.nodes[idx].rssi_value,
            "sf": consts.nodes[idx].sf_value
        }

    def render(self):
        if self.render_mode == 'human':
            print({agent: self.observe(agent) for agent in self.agents})

    def _calculate_reward(self, agent_index):
        lambda_value = 0.0001
        mean_prr = consts.nodes[agent_index].calculate_prr()
        retransmission_penalty = lambda_value * consts.nodes[agent_index].packets_sent_count
        return mean_prr - retransmission_penalty

# To support the AEC API from this parallel environment
def env_creator(env_config):
    env = LoRaEnvParallel(**env_config)
    env = parallel_to_aec(env)
    return env

# Example usage
env = LoRaEnvParallel(render_mode="human")
observations = env.reset()
while True:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, dones, _, infos = env.step(actions)
    if all(dones.values()):
        break
env.close()
