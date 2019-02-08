import statistics
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class RolloutDataSetAbstract(Dataset):
    def __init__(self, env_config):
        """

        :param discount_factor:
        :param transform: transform to apply to observation
        """
        super().__init__()
        self.episode = []
        self.rollout = []
        self.value = []
        self.start = 0
        self.discount_factor = env_config.discount_factor
        self.transform = env_config.transform

    def append(self, observation, reward, action, done):
        self.rollout.append((observation, reward, action))
        self.advantage(observation, reward, action, done)

    def advantage(self, observation, reward, action, done):
        raise NotImplementedError

    def normalize(self):
        mean = statistics.mean(self.value)
        stdev = statistics.stdev(self.value)
        self.value = [(vl - mean) / (stdev + 1e-12) for vl in self.value]

    def total_reward(self):
        return sum([reward[1] for reward in self.rollout])

    def transform(self, observation, insert_batch=False):
        raise NotImplementedError

    def __getitem__(self, item):
        observation, reward, action = self.rollout[item]
        value = self.value[item]
        observation_t = self.transform(observation)
        return observation_t, reward, action, value

    def __len__(self):
        return len(self.rollout)


class RolloutDataSet(RolloutDataSetAbstract):
    def __init__(self, discount_factor):
        super().__init__(discount_factor)

    def advantage(self, observation, reward, action, done):
        if done:
            values = []
            cum_value = 0.0
            # calculate values
            for step in reversed(range(self.start, len(self.rollout))):
                cum_value = self.rollout[step][1] + cum_value * self.discount_factor
                values.append(cum_value)
            self.value = self.value + list(reversed(values))
            self.start = len(self.rollout)

class PongDataset(RolloutDataSetAbstract):
    def __init__(self, discount_factor, features):
        super().__init__(discount_factor)
        self.features = features

    def advantage(self, observation, reward, action, done):
        if reward != 0.0:
            values = []
            cum_value = 0.0
            # calculate values
            for step in reversed(range(self.start, len(self.rollout))):
                cum_value = self.rollout[step][1] + cum_value * self.discount_factor
                values.append(cum_value)
            self.value = self.value + list(reversed(values))
            self.start = len(self.rollout)


class Step:
    def __init__(self, observation, action, reward, done):
        self.observation = observation
        self.reward = reward
        self.action = action
        self.advantage = None
        self.done = done

    def as_tuple(self):
        return self.observation, self.action, self.reward, self.advantage


class BufferedRolloutDataset(Dataset):
    def __init__(self, discount_factor, transform=None):
        """
        BufferedRolloutDataset can collect multiple episodes in parallel, useful for multi-agent sims
        :param discount_factor:
        :param transform: transform applied to the observations,
        if for example you want to return observation as a tensor
        """
        super().__init__()
        self.buffer = {}
        self.rollouts = []
        self.discount_factor = discount_factor
        self.transform = transform if transform else self.default_transform

    def append(self, observation, action, reward, done, episode=0):
        """

        :param observation:
        :param action:
        :param reward:
        :param done: set this flag to save as an episode, will assign the discounted rewards
        :param episode: used to collect multiple episodes in parallel, for multi-agent simulations,
        just set episode to a unique id, id's can be re-used after append is called with "done" flag set True
        """
        if episode in self.buffer:
            self.buffer[episode].append(Step(observation, action, reward, done))
        else:
            self.buffer[episode] = []
            self.buffer[episode].append(Step(observation, action, reward, done))

        if done:
            self._end_episode(episode)
            del self.buffer[episode]

    def _end_episode(self, episode):
        cum_value = 0.0
        for step in reversed(self.buffer[episode]):
            cum_value = step.reward + cum_value * self.discount_factor
            step.advantage = cum_value
        for step in self.buffer[episode]:
            self.rollouts.append(step)

    def end_rollout(self):
        """
        End rollout normalizes the episode rewards, call at the end of the rollout
        """
        all_advantages = [step.advantage for step in self.rollouts]
        mean = statistics.mean(all_advantages)
        stdev = statistics.stdev(all_advantages)
        for step in self.rollouts:
            step.advantage = (step.advantage - mean) / (stdev + 1e-12)

    def total_reward(self):
        return sum([step.reward for step in self.rollouts])

    def default_transform(self, observation, insert_batch=False):
        """
        Default is to assume input is a numpy array and output is a pytorch tensor,
        :param observation: assumed to be a numpy array
        :param insert_batch: inserts a batch dimension
        :return: a pytorch tensor
        """

        return torch.from_numpy(observation).unsqueeze(0) if insert_batch else torch.from_numpy(observation)

    def __getitem__(self, item):
        step = self.rollouts[item]
        observation_t = self.transform(step.observation)
        return observation_t, step.reward, step.action, step.advantage

    def __len__(self):
        return len(self.rollouts)