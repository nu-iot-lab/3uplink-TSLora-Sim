import gym
from gym import spaces
import numpy as np


class LoRaEnv(gym.Env):
    """Custom Environment for Time-Slotted 3-uplink LoRa network simulation that follows gym interface"""
    metadata = {'render.modes': ['console']}

    def __init__(self, num_agents, data_size, avg_wake_up_time, sim_time):
        super(LoRaEnv, self).__init__()
        
        self.num_agents = num_agents
        self.data_size = data_size
        self.avg_wake_up_time = avg_wake_up_time
        self.sim_time = sim_time
        
        # Action space is now choosing between 1 to 3 retransmissions
        self.action_space = spaces.Discrete(3)  # 0 for 1 retransmission, 1 for 2, 2 for 3
        
        # Observation space: [distance/SF, power, success rate]
        # Note: Adjust the distance range according to your network's scale
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([10, 1, 1]), dtype=np.float32)
        
        # Initialize state
        self.state = None
        self.reset()

    def reset(self):
        """Resets the state of the environment and returns an initial observation."""
        # Randomly initialize each agent's state
        self.state = np.random.uniform(low=self.observation_space.low, high=self.observation_space.high)
        return self.state

    def step(self, action):
        """Executes a step in the environment."""
        distance, _, success_rate = self.state
        
        # Convert action to actual number of retransmissions
        retransmissions = action + 1  # Action is 0, 1, or 2; retransmissions is 1, 2, or 3
        
        # Update success rate based on distance and retransmissions
        # Closer nodes with fewer retransmissions may have a similar success rate to farther nodes with more
        if distance < 3:  # Assuming distance is scaled such that 3 represents close proximity
            success_rate += 0.05 * retransmissions
        else:
            success_rate += 0.03 * retransmissions
        
        success_rate = min(success_rate, 1.0)  # Cap the success rate at 1
        
        # Reward: Higher for higher success rates, adjusted for number of retransmissions
        reward = success_rate - 0.1 * retransmissions
        
        # Update the state with new success rate
        self.state[2] = success_rate
        
        # Simulate time passing in the simulation
        self.sim_time -= 1
        
        # Check if simulation is done
        done = self.sim_time <= 0
        
        return self.state, reward, done, {}

    def render(self, mode='console'):
        """Renders the environment."""
        if mode != 'console':
            raise NotImplementedError()
        print(f"State: {self.state}, Time left: {self.sim_time}")

    def close(self):
        """Perform any necessary cleanup."""
        pass
