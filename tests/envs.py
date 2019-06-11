import gym
from gym import Env
from gym.spaces import Discrete
import numpy as np


class LineWalk(Env):
    def __init__(self):
        super().__init__()
        self.empty_obs = np.zeros((4,), dtype=np.float32)
        self.observation_space = self.reset()
        self.action_space = Discrete(2)
        self.position = 0
        self.action_vector = [-1, 1]

    def reset(self):
        self.observation_space = np.copy(self.empty_obs)
        self.position = 0
        self.observation_space[self.position] = 1.0
        return self.observation_space

    def step(self, action):
        vector = self.action_vector[action]
        next_pos = self.position + vector

        if 0 <= next_pos <= 3:
            self.position = next_pos

        self.observation_space = np.copy(self.empty_obs)
        self.observation_space[self.position] = 1.0

        reward = 0.0
        done = False
        if self.position == 3:
            reward = 1.0
            done = True

        return self.observation_space, reward, done, {}


gym.register('LineWalk-v0', entry_point='tests.envs:LineWalk', max_episode_steps=8)


class Bandit(Env):
    def __init__(self):
        super().__init__()
        self.observation_space = np.zeros((3,), dtype=np.float32)
        self.action_space = Discrete(2)
        self.position = 1
        self.action_vector = [-1, 1]

    def reset(self):
        self.observation_space.fill(0)
        self.position = 1
        self.observation_space[self.position] = 1.0
        return self.observation_space

    def step(self, action):
        vector = self.action_vector[action]
        next_pos = self.position + vector

        if 0 <= next_pos <= 2:
            self.position = next_pos

        self.observation_space.fill(0.0)
        self.observation_space[self.position] = 1.0

        reward = 0.0
        done = False
        if self.position == 2:
            reward = 1.0
            done = True

        if self.position == 0:
            reward = -1.0
            done = True

        return self.observation_space, reward, done, {}


gym.register('Bandit-v0', entry_point='tests.envs:Bandit')