import statistics
from abc import ABCMeta, abstractmethod

import torch
import numpy as np
from torch.utils.data import Dataset
import redis
import struct
from bisect import bisect_right


class RolloutDatasetAbstract(Dataset, metaclass=ABCMeta):
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

    @abstractmethod
    def advantage(self, observation, reward, action, done):
        pass

    def normalize(self):
        mean = statistics.mean(self.value)
        stdev = statistics.stdev(self.value)
        self.value = [(vl - mean) / (stdev + 1e-12) for vl in self.value]

    def total_reward(self):
        return sum([reward[1] for reward in self.rollout])

    @abstractmethod
    def transform(self, observation, insert_batch):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def __len__(self):
        pass


class RedisDataset(RolloutDatasetAbstract):

    def __init__(self, host='localhost', port=6379, db=0):
        self.db = Db(host, port, db)
        self.rollout = None

    def __getitem__(self, item):
        pass


class SingleProcessDataSetAbstract(RolloutDatasetAbstract):

    def append(self, observation, reward, action, done):
        self.rollout.append((observation, reward, action))
        self.advantage(observation, reward, action, done)

    def advantage(self, observation, reward, action, done):
        raise NotImplementedError

    def transform(self, observation, insert_batch=False):
        raise NotImplementedError

    def __getitem__(self, item):
        observation, reward, action = self.rollout[item]
        value = self.value[item]
        observation_t = self.transform(observation)
        return observation_t, reward, action, value

    def __len__(self):
        return len(self.rollout)


class SingleProcessDataSet(SingleProcessDataSetAbstract):
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


class PongDataset(SingleProcessDataSetAbstract):
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


class Db:
    def __init__(self, host='localhost', port=6379, db=0):
        self.db = redis.Redis(host=host, port=port, db=db)

    def create_rollout(self, observation_dtype):
        if self.db.get('rollouts') is None:
            self.db.set('rollouts', '0')
        rollout = Rollout(self.db, int(self.db.get('rollouts')), observation_dtype)
        self.db.incr('rollouts')
        return rollout

    def drop(self):
        """clears all data from the database"""
        self.db.flushall()

class Rollout:
    def __init__(self, db, id, observation_dtype):
        self.id = id
        self.db = db
        self.episode_len = []
        self.episode_off = []
        self.observation_dtype = observation_dtype

    def end(self):
        len = int(self.db.get(self.key()))
        for episode in range(len):
            self.episode_len.append(int(self.db.llen(Episode.key(self.id, episode))))

        offset = 0
        for l in self.episode_len:
            self.episode_off.append(offset)
            offset += l

    def start(self):
        pass

    def ready(self):
        return True

    def delete(self):
        return True

    def get_dataset(self):
        pass

    def key(self):
        return f'r{self.id}'

    def create_episode(self):
        if self.db.get(self.key()) is None:
            self.db.set(self.key(), '0')
        episode = Episode(self, self.db, int(self.db.get(self.key())))
        self.db.incr(self.key())
        return episode

    @staticmethod
    def find_le(a, x):
        'Find rightmost value less than or equal to x'
        i = bisect_right(a, x)
        if i:
            return i - 1
        raise ValueError

    def __getitem__(self, item):
        episode_id = Rollout.find_le(self.episode_off, item)
        step_i = item - self.episode_off[episode_id]
        encoded_step = self.db.lindex(Episode.key(self.id, episode_id), step_i)
        step = Step.decode(encoded_step, self.observation_dtype)
        return step

    def __len__(self):
        return sum(self.episode_len)


class Episode:
    def __init__(self, rollout, db, id):
        self.rollout = rollout
        self.db = rollout, db
        self.id = id

    def end(self):
        pass

    @staticmethod
    def key(rollout_id, episode_id):
        return f'r{rollout_id}_e{episode_id}'

    def append(self, step):
        self.rollout.db.lpush(Episode.key(self.rollout.id, self.id), step.encode())


class Step:
    def __init__(self, observation, action, reward, done):
        self.observation = observation
        self.reward = reward
        self.action = action
        self.advantage = None
        self.done = done

    def as_tuple(self):
        return self.observation, self.action, self.reward, self.advantage

    def encode(self):
        """ encode step to bstring"""
        o_h, o_w = self.observation.shape
        b_ard = struct.pack('>if?II', self.action, self.reward, self.done, o_h, o_w)
        b_o = self.observation.tobytes()
        encoded = b_ard + b_o
        return encoded

    @staticmethod
    def decode(encoded, obs_dtype):
        """ decode step from bstring"""
        len_bard = struct.calcsize('>if?II')
        action, reward, done, o_h, o_w = struct.unpack('>if?II', encoded[:len_bard])
        o = np.frombuffer(encoded, dtype=obs_dtype, offset=len_bard).reshape(o_h, o_w)
        return Step(o, action, reward, done)


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


def toRedis(r, a, n):
    """Store given Numpy array 'a' in Redis under key 'n'"""
    h, w = a.shape
    shape = struct.pack('>II', h, w)
    encoded = shape + a.tobytes()

    # Store encoded data in Redis
    r.set(n, encoded)
    return


def fromRedis(r, n):
    """Retrieve Numpy array from Redis key 'n'"""
    encoded = r.get(n)
    h, w = struct.unpack('>II', encoded[:8])
    a = np.frombuffer(encoded, dtype=np.uint32, offset=8).reshape(h, w)
    return a
