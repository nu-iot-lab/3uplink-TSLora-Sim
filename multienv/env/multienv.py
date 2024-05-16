import simpy
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec
from simulator.lora_simulator import LoraSimulator
import simulator.consts as consts
import simulator.utils as utils
from copy import copy
import functools
import logging


class LoRaEnvParallel(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "multiagent_v0"}

    def __init__(
        self,
        nodes_count=10,
        data_size=16,
        avg_wake_up_time=30,
        sim_time=3600,
        render_mode=None,
    ):
        self.possible_agents = [f"agent_{i}" for i in range(nodes_count)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, range(len(self.possible_agents)))
        )
        self.render_mode = render_mode
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

        self.current_step = 0
        self.done = False
        self.truncated = False

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Dict(
            {
                "prr": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
                "rssi": spaces.Box(low=-200.0, high=0.0, shape=(), dtype=np.float32),
                "sf": spaces.Box(low=7, high=12, shape=(), dtype=np.int64),
            }
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(3)

    def reset(self, **kwargs):
        logging.info("Resetting environment.")
        self.agents = copy(self.possible_agents)
        self.simpy_env = simpy.Environment()
        self.simulator = LoraSimulator(
            self.nodes_count,
            self.data_size,
            self.avg_wake_up_time * 1000,
            self.sim_time * 1000,
            self.simpy_env,
        )
        self.current_step = 0
        self.done = False
        self.truncated = False
        utils.reset_simulator()
        self.simulator.add_nodes()
        observations = {agent: self.observe(agent) for agent in self.possible_agents}
        infos = {agent: {} for agent in self.possible_agents}
        logging.info("Environment reset complete.")
        return observations, infos

    def step(self, actions):
        logging.info(f"Step {self.current_step} with actions {actions}.")
        if self.current_step == 0:
            self.simulator.start_simulation()
        if self.current_step >= self.sim_time:
            self.done = True
            reward = self._calculate_reward()
            observations = {
                agent: self.observe(agent) for agent in self.possible_agents
            }
            infos = {agent: {} for agent in self.possible_agents}
            logging.info("Simulation done.")
            return observations, reward, self.done, infos

        for agent in actions:
            agent_index = self.agent_name_mapping[agent]
            if not self.done:
                self.simulator.update_nodes_behavior(agent_index, actions[agent])

        self.current_step += 1
        timestep = self.current_step * 1000
        for i in range(self.nodes_count):
            utils.log(
                f"!-- UPLINK NUMBER FOR STEP [{self.current_step}] FOR NODE {i}: {actions[f'agent_{i}'] + 1} --!",
                self.simpy_env,
            )
        self.simulator.env.run(until=timestep)

        observations = {agent: self.observe(agent) for agent in self.possible_agents}
        rewards = {
            agent: self._calculate_reward(self.agent_name_mapping[agent])
            for agent in self.possible_agents
        }
        self.done = self.current_step >= self.sim_time
        infos = {agent: {} for agent in self.possible_agents}

        if self.render_mode == "human":
            self.render()

        dones = {agent: self.done for agent in self.possible_agents}
        truncations = {agent: self.truncated for agent in self.possible_agents}

        return observations, rewards, dones, truncations, infos

    def observe(self, agent):
        idx = self.agent_name_mapping[agent]
        return {
            "prr": np.array(consts.nodes[idx].prr_value, dtype=np.float32),
            "rssi": np.array(consts.nodes[idx].rssi_value, dtype=np.float32),
            "sf": np.array(consts.nodes[idx].sf_value, dtype=np.int64),
        }

    def render(self):
        if self.render_mode == "human":
            print({agent: self.observe(agent) for agent in self.possible_agents})

    def _calculate_reward(self, agent_index):
        lambda_value = 0.0001
        mean_prr = consts.nodes[agent_index].calculate_prr()
        retransmission_penalty = (
            lambda_value * consts.nodes[agent_index].packets_sent_count
        )
        return mean_prr - retransmission_penalty


def env_creator(env_config):
    env_instance = LoRaEnvParallel(**env_config)
    env_instance = parallel_to_aec(env_instance)
    return env_instance
