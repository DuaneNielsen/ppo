import gym
from gym import Env
from gym.spaces import Discrete
import numpy as np
import torch
import struct

"""

  State transition diagram, can move in both directions

  S | 0 | 0 | T(+1)
"""


class LineWalk(Env):
    def __init__(self):
        super().__init__()
        self.length = 10
        self.action_space = Discrete(2)
        self.position = 0
        self.action_vector = [-1, 1]

    def empty_obs(self):
        return np.zeros((self.length,), dtype=np.float32)

    def reset(self, state=None):
        self.observation_space = self.empty_obs()
        self.position = 0
        self.observation_space[self.position] = 1.0
        return self.observation_space

    def step(self, action):
        vector = self.action_vector[action]
        next_pos = self.position + vector

        if 0 <= next_pos <= self.length-1:
            self.position = next_pos

        self.observation_space = self.empty_obs()
        self.observation_space[self.position] = 1.0

        reward = 0.0
        done = False
        if self.position == self.length -1:
            reward = 1.0
            done = True

        return self.observation_space, reward, done, {}

    def render(self, mode='human'):
        print(self.observation_space)

    def __getstate__(self):
        return struct.pack('i', self.position)

    def __setstate__(self, state):
        self.__init__()
        self.position = struct.unpack('i', state)[0]


gym.register('LineWalk-v0', entry_point='tests.envs:LineWalk', max_episode_steps=200)

"""

    Simple bandit environment... move left to get punished, move right to get rewarded

  T -1 | S 0 | T +1

"""


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

    def render(self, mode='human'):
        print(self.observation_space)

    def __getstate__(self):
        return struct.pack('i', self.position)

    def __setstate__(self, state):
        self.__init__()
        self.position = struct.unpack('i', state)[0]


gym.register('Bandit-v0', entry_point='tests.envs:Bandit')

"""

    Simple bandit environment... move left to get punished, move right to get rewarded

  T -1 | S 0 | T +1
  
  state is a numpy array of shape (3,3)
  
    row 1 -> state in position           [ 0, 1, 0 ]
    row 2 -> state if action 0 is taken  [ 1, 0, 0 ]
    row 3 -> state if action 1 is taken  [ 0, 0, 1 ]

"""


class BanditLookahead(Env):
    def __init__(self):
        super().__init__()
        self.observation_space = np.zeros((3, 5,), dtype=np.float32)
        self.action_space = Discrete(2)
        self.position = 2
        self.action_vector = [-1, 1]

    def reset(self):
        self.observation_space.fill(0)
        self.position = 2
        self.observation_space[0, 2] = 1.0
        self.observation_space[1, 1] = 1.0
        self.observation_space[2, 3] = 1.0
        return self.observation_space

    def next_pos(self, position, vector):
        next_pos = self.position + vector
        if 0 <= next_pos <= 4:
            position = next_pos
        return position

    def step(self, action):
        vector = self.action_vector[action]
        self.position = self.next_pos(self.position, vector)
        left_pos = self.next_pos(self.position, -1)
        right_pos = self.next_pos(self.position, 1)

        self.observation_space.fill(0.0)
        self.observation_space[0, self.position] = 1.0
        self.observation_space[1, left_pos] = 1.0
        self.observation_space[2, right_pos] = 1.0

        reward = 0.0
        done = False
        if self.position == 4:
            reward = 1.0
            done = True

        if self.position == 0:
            reward = -1.0
            done = True

        return self.observation_space, reward, done, {}

    def render(self, mode='human'):
        print(self.observation_space)


gym.register('BanditLookahead-v0', entry_point='tests.envs:BanditLookahead')


class BanditLookaheadTransform:
    def __call__(self, state, dtype):
        current_position = state[0]
        return torch.from_numpy(current_position).to(dtype=dtype)

"""

    Simple bandit environment... move left to get punished, move right to get rewarded

  T -1 | S 0 | T +1

"""


class BanditLookAhead_v1(Env):
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

    def render(self, mode='human'):
        print(self.observation_space)

    def __getstate__(self):
        return struct.pack('i', self.position)

    def __setstate__(self, state):
        self.position = struct.unpack('i', state)



gym.register('BanditLookahead-v1', entry_point='tests.envs:BanditLookAhead_v1')