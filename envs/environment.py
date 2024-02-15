import gym
from gym import spaces
import numpy as np

class SimulatorEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,))
        self.state = np.array([0, 0])
        self.steps_beyond_done = None

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        if action == 0:
            state[0] += 1
        else:
            state[1] += 1
        self.state = state
        done = bool(state[0] > 10)
        reward = 1 if done else 0
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = np.array([0, 0])
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass