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
import matplotlib.pyplot as plt

# how to run?
# python3 sarsa.py


class SARSAAgent:
    def __init__(self, action_space, observation_space, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        self.action_space = action_space
        self.observation_space = observation_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}
        self.retransmissions = 0  # Counter for retransmissions

    def get_state_key(self, observation):
        # Convert the observation dictionary to a tuple of values
        return tuple(observation[key].item() for key in sorted(observation))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space.n)
        return np.argmax(self.q_table[state_key])

    def update(self, state, action, reward, next_state, next_action):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space.n)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_space.n)
        td_target = reward + self.discount_factor * self.q_table[next_state_key][next_action]
        td_error = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.learning_rate * td_error

    def reset_retransmissions(self):
        self.retransmissions = 0

    def add_retransmission(self, num_retransmissions):
        self.retransmissions += num_retransmissions

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
                agents[agent].add_retransmission(actions[agent] + 1)  # Record retransmissions

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

        # def self._calculate_reward(self):
        #     lambda_value = 0.0001
        #     mean_prr = np.mean([node.calculate_prr() for node in consts.nodes])
        #     retransmission_penalty = lambda_value * sum(
        #         [node.packets_sent_count for node in consts.nodes]
        #     )
        #     return mean_prr - retransmission_penalty

        dones = {agent: self.done for agent in self.possible_agents}
        truncations = {agent: self.truncated for agent in self.possible_agents}

        return observations, rewards, dones, truncations, infos

    def observe(self, agent):
        idx = self.agent_name_mapping[agent]
        return {
            "prr": np.array(consts.nodes[idx].calculate_prr(), dtype=np.float32),
            "rssi": np.array(consts.nodes[idx].rssi_value, dtype=np.float32),
            "sf": np.array(consts.nodes[idx].sf_value, dtype=np.int64),
        }

    def render(self):
        if self.render_mode == "human":
            print({agent: self.observe(agent) for agent in self.possible_agents})

    def _calculate_reward(self, agent_index):
        lambda_value = 0.0001
        mean_prr = consts.nodes[agent_index].calculate_prr()
        retransmission_penalty = lambda_value * consts.nodes[agent_index].packets_sent_count
        return mean_prr - retransmission_penalty

# Initialize the environment and agent
env_config = {
    "nodes_count": 10,
    "data_size": 16,
    "avg_wake_up_time": 30,
    "sim_time": 3600
}
env = LoRaEnvParallel(**env_config)
agents = {agent: SARSAAgent(env.action_space(agent), env.observation_space(agent)) for agent in env.possible_agents}

# Training loop
episode_rewards = []
episode_retransmissions = []
num_episodes = 100

for episode in range(num_episodes):
    observations, infos = env.reset()
    actions = {agent: agents[agent].choose_action(observations[agent]) for agent in env.possible_agents}
    
    done = False
    total_rewards = {agent: 0 for agent in env.possible_agents}
    total_retransmissions = {agent: 0 for agent in env.possible_agents}
    for agent in env.possible_agents:
        agents[agent].reset_retransmissions()  # Reset retransmission counter
    while not done:
        next_observations, rewards, dones, truncations, infos = env.step(actions)
        next_actions = {agent: agents[agent].choose_action(next_observations[agent]) for agent in env.possible_agents}
        for agent in env.possible_agents:
            agents[agent].update(observations[agent], actions[agent], rewards[agent], next_observations[agent], next_actions[agent])
            total_rewards[agent] += rewards[agent]
            total_retransmissions[agent] += agents[agent].retransmissions
        observations = next_observations
        actions = next_actions
        done = all(dones.values())

    episode_total_reward = sum(total_rewards.values())
    episode_total_retransmissions = sum(total_retransmissions.values())
    episode_rewards.append(episode_total_reward)
    episode_retransmissions.append(episode_total_retransmissions)
    logging.info(f"Episode {episode} complete with total reward: {episode_total_reward} and total retransmissions: {episode_total_retransmissions}")

# Plot the learning curve for total rewards
plt.figure(figsize=(12, 5))

# Plot rewards per episode
plt.subplot(1, 2, 1)
plt.plot(episode_rewards, label='Total Reward')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('SARSA Training - Total Reward per Episode')
plt.legend()

# Plot retransmissions per episode
plt.subplot(1, 2, 2)
plt.plot(episode_retransmissions, label='Total Retransmissions', color='red')
plt.xlabel('Episode')
plt.ylabel('Total Retransmissions')
plt.title('SARSA Training - Total Retransmissions per Episode')
plt.legend()

plt.tight_layout()
plt.show()
