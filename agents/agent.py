import numpy as np

from envs.environment import LoRaEnv

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_rate=0.95, exploration_rate=1.0, exploration_decay=0.99, exploration_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self.q_table = np.zeros((state_size, action_size))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.action_size)
        return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state, done):
        if not done:
            max_future_q = np.max(self.q_table[next_state])
            current_q = self.q_table[state, action]
            new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_rate * max_future_q)
            self.q_table[state, action] = new_q
        else:
            self.q_table[state, action] = (1 - self.learning_rate) * self.q_table[state, action] + self.learning_rate * reward
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

# Example usage within an RL loop
env = LoRaEnv(num_agents=1, data_size=16, avg_wake_up_time=30, sim_time=7200)
agent = QLearningAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n)

for episode in range(1000):  # Number of episodes
    state = env.reset()
    state = np.reshape(state, [1, env.state_size])
    total_reward = 0

    while True:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, env.state_size])
        
        agent.update_q_table(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    print(f"Episode: {episode+1}, Total reward: {total_reward}, Exploration rate: {agent.exploration_rate}")
