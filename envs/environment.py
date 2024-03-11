import gymnasium as gym
import numpy as np

from lora_simulator import LoraSimulator


class LoRaEnv(gym.Env):
    metadata = {'render_modes': ['console']}

    def __init__(self, num_agents, data_size, avg_wake_up_time, sim_time):
        super(LoRaEnv, self).__init__()

        LoraSimulator(nodes_count=num_agents, data_size=data_size, avg_wake_up_time=avg_wake_up_time, sim_time=sim_time)
        
        self.num_agents = num_agents
        self.data_size = data_size
        self.avg_wake_up_time = avg_wake_up_time
        self.sim_time = sim_time
        # self.sf = sf  # Spreading factor, affects the effectiveness of retransmissions
        self.lambda_value = 0.1  # Weight for penalizing retransmissions
        self.sf = 7
        
        # Observation space (RSSI, PRR)
        self.observation_space = gym.spaces.Box(low=np.array([-120, 0]), high=np.array([0, 1]), dtype=np.float64)
        
        # Action space is choosing the number of retransmissions (1 to 3)
        self.action_space = gym.spaces.Discrete(3)  # 0: 1 retransmission, 1: 2 retransmissions, 2: 3 retransmissions
        
        self.state = None
        self.total_prr = 0  # Keep track of the total PRR
        self.total_retransmissions = 0  # Keep track of the total number of retransmissions

    def step(self, action):
        # Simplified model for state update
        sf_effect = 1 - (self.sf - 7) / 6
        rssi_change = np.random.uniform(-5, 5)
        prr_bonus = (action + 1) * 0.1 * sf_effect  # Effectiveness of retransmissions scaled by SF
        
        new_rssi = self.state[0] + rssi_change
        new_prr = min(self.state[1] + prr_bonus, 1)  # Ensure PRR doesn't exceed 1
        
        self.state = np.array([new_rssi, new_prr])
        self.total_prr += new_prr  # Update total PRR
        self.total_retransmissions += (action + 1)  # Update total retransmissions (action + 1 because actions are 0-indexed)

        # Reward calculation
        reward = self.total_prr - self.lambda_value * self.total_retransmissions
        
        terminated = False  # This could be more complex based on your simulation needs
        truncated = False
        info = {'total_prr': self.total_prr, 'total_retransmissions': self.total_retransmissions}
        
        return self.state, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset the environment state and tracking variables
        initial_rssi = np.random.uniform(-120, 0)
        initial_prr = np.random.uniform(0, 1)
        
        self.state = np.array([initial_rssi, initial_prr])
        self.total_prr = 0
        self.total_retransmissions = 0

        return self.state, {}
    
    def render(self, mode='human'):
        # Print the current state
        print(f"State: RSSI={self.state[0]:.2f} dBm, PRR={self.state[1]:.2f}, SF={self.sf}")

# Example of creating and testing the environment with a specific penalty coefficient
# env = LoRaEnv(sf=7, )
env = LoRaEnv(num_agents=10, data_size=100, avg_wake_up_time=5, sim_time=100)
initial_state = env.reset()
# print(f"Initial State: RSSI={initial_state[0]:.2f} dBm, PRR={initial_state[1]:.2f}")