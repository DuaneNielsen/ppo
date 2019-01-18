import statistics
import threading

import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

class RolloutDataSet(Dataset):
    def __init__(self, discount_factor, transform):
        """

        :param discount_factor:
        :param transform: transform to apply to observation
        """
        super().__init__()
        self.rollout = []
        self.value = []
        self.start = 0
        self.discount_factor = discount_factor
        self.transform = transform

    def append(self, observation, reward, action, done):
        self.rollout.append((observation, reward, action))
        if reward != 0.0:
            self.end_game()

    def end_game(self):
        values = []
        cum_value = 0.0
        # calculate values
        for step in reversed(range(self.start, len(self.rollout))):
            cum_value = self.rollout[step][1] + cum_value * self.discount_factor
            values.append(cum_value)
        self.value = self.value + list(reversed(values))
        self.start = len(self.rollout)

    def normalize(self):
        mean = statistics.mean(self.value)
        stdev = statistics.stdev(self.value)
        self.value = [(vl - mean) / stdev for vl in self.value]

    def total_reward(self):
        return sum([reward[1] for reward in self.rollout])

    def __getitem__(self, item):
        observation, reward, action = self.rollout[item]
        value = self.value[item]
        #observation_t = to_tensor(np.expand_dims(observation, axis=2))
        observation_t = self.transform(observation)
        return observation_t, reward, action, value

    def __len__(self):
        return len(self.rollout)