import gym
import numpy as np


class OneHotObsWrapper(gym.ObservationWrapper):
    """
    converts discrete observations to one-hot vectors
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space.shape = (self.env.observation_space.n, )

    def observation(self, observation):
        one_hot = np.zeros(self.env.observation_space.n, dtype=np.float)
        one_hot[observation] = 1.0
        return one_hot