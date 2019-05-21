import statistics
from abc import ABCMeta, abstractmethod

import torch
import numpy as np
from torch.utils.data import Dataset
import redis
import struct
from bisect import bisect_right, bisect_left
import uuid
from redis.lock import Lock
import logging
import duallog


class RedisSequence:
    """
    Connects and manages sequences across threads
    """

    def __init__(self, redis, key, reset=False):
        self.redis = redis
        self.key = key
        if self.redis.get(key) is None or reset:
            self.redis.set(key, 0)

    def __iter__(self):
        return self

    def __next__(self):
        p = self.redis.pipeline()
        p = p.get(self.key)
        p.incr(self.key)
        result = p.execute()
        return int(result[0])

    def current(self):
        return int(self.redis.get(self.key)) - 1


class RolloutDatasetAbstract(Dataset, metaclass=ABCMeta):
    def __init__(self):
        """

        :param discount_factor:
        :param transform: transform to apply to observation
        """
        super().__init__()

    @abstractmethod
    def advantage(self, episode):
        pass

    @abstractmethod
    def normalize(self):
        pass

    @abstractmethod
    def total_reward(self):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def __len__(self):
        pass


class RolloutDatasetBase(RolloutDatasetAbstract):
    def __init__(self, env_config, rollout):
        """

        :param discount_factor:
        :param transform: transform to apply to observation
        """
        super().__init__()
        self.rollout = rollout
        self.rollout.finalize()
        self.adv = [0.0 for _ in range(len(rollout))]
        self.start = 0
        self.discount_factor = env_config.discount_factor
        self.transform = env_config.transform

        # calculate advantage values
        for episode in rollout:
            self.advantage(episode)
        self.normalize()

    def normalize(self):
        mean = statistics.mean(self.adv)
        stdev = statistics.stdev(self.adv)
        self.adv = [(vl - mean) / (stdev + 1e-12) for vl in self.adv]

    def advantage(self, episode):

        cum_value = 0.0
        offset = self.rollout.offset(episode)

        for step in reversed(range(offset, offset + len(episode))):
            cum_value = self.rollout[step].reward + cum_value * self.discount_factor
            self.adv[step] = cum_value

    def total_reward(self):
        reward = 0.0
        for episode in self.rollout:
            for step in episode:
                reward += step.reward
        return reward

    def __getitem__(self, item):
        step = self.rollout[item]
        advantage = self.adv[item]
        t_obs = self.transform(step.observation)
        return t_obs, step.action, step.reward, advantage

    def __len__(self):
        return len(self.rollout)


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
        value = self.adv[item]
        observation_t = self.transform(observation)
        return observation_t, reward, action, value

    def __len__(self):
        return len(self.rollout)


class SingleProcessDataSet(SingleProcessDataSetAbstract):
    def __init__(self, discount_factor):
        super().__init__(discount_factor)


# class PongDataset(SingleProcessDataSetAbstract):
#     def __init__(self, discount_factor, features):
#         super().__init__(discount_factor)
#         self.features = features
#
#     def advantage(self, observation, reward, action, done):
#         if reward != 0.0:
#             values = []
#             cum_value = 0.0
#             # calculate values
#             for step in reversed(range(self.start, len(self.episode))):
#                 cum_value = self.episode[step].reward + cum_value * self.discount_factor
#                 values.append(cum_value)
#             self.value = self.value + list(reversed(values))
#             self.start = len(self.rollout)

# todo I think these might be better as static methods on the RedisRollout class

class Db:
    def __init__(self, host='localhost', port=6379, password=None, db=0, redis_client=None):
        if redis_client is None:
            self.redis = redis.Redis(host=host, port=port, db=db, password=password)
        else:
            self.redis = redis_client
        self.rollout_seq = RedisSequence(self.redis, 'rollout')

    def drop(self):
        """clears all data from the database"""
        self.redis.flushall()

    def create_rollout(self, env_config):
        return RedisRollout(self.redis, env_config, next(self.rollout_seq))

    def latest_rollout(self, env_config):
        return RedisRollout(self.redis, env_config, self.rollout_seq.current())

    def rollout(self, id, config):
        return RedisRollout(self.redis, config, id)

    def delete_rollout(self, rollout):
        # todo batch delete
        for key in self.redis.scan_iter(f'rollout-{rollout.id}*'):
            self.redis.delete(key)

        self.redis.delete(rollout.key())

    def clear_rollouts(self):
        for key in self.redis.scan_iter("rollout-*"):
            self.redis.delete(key)


class RedisRollout:
    def __init__(self, redis, env_config, id):
        self.id = id
        self.redis = redis
        self.episodes = []
        self.episode_len = []
        self.episode_off = []
        self.len = 0
        self.env_config = env_config

    def finalize(self):
        """
        Call before you generate a dataset
        This caches all the guids in the db at time of calling to a list
        creating a data structure which allows for fast access by the dataset object
        and fixing the length of the the dataset
        """

        lockname = self.key('lock')
        lk = Lock(self.redis, lockname)
        logging.debug(f'getting lock {lockname}')
        lk.acquire()
        logging.debug(f'getting lock {lockname}')
        self.redis.set(self.key('finalized'), 'FINALIZED')
        lk.release()
        logging.debug(f'released lock {lockname}')

        self.episodes = []
        self.episode_len = []
        for episode in range(self.num_episodes()):
            self.episodes.append(self.redis.lindex(self.key('episodes'), episode).decode())

        self.episodes = sorted(self.episodes)

        for episode_id in self.episodes:
            self.episode_len.append(len(Episode(self, self.redis, episode_id)))

        self.episode_off = []
        offset = 0
        for l in self.episode_len:
            self.episode_off.append(offset)
            offset += l

    def create_episode(self):
        return Episode(self, self.redis, self.key(str(uuid.uuid4())))

    def key(self, subkey=''):
        if subkey == '':
            return f'rollout-{self.id}'
        return f'rollout-{self.id}-' + subkey

    @staticmethod
    def find_le(a, x):
        'Find rightmost value less than or equal to x'
        i = bisect_right(a, x)
        if i:
            return i - 1
        raise ValueError

    @staticmethod
    def index(a, x):
        'Locate the leftmost value exactly equal to x'
        i = bisect_left(a, x)
        if i != len(a) and a[i] == x:
            return i
        raise ValueError

    def num_episodes(self):
        """ number of episodes in a rollout """
        return self.redis.llen(self.key('episodes'))

    def get_index_for_episode(self, episode):
        if self.episodes is None:
            raise Exception
        return RedisRollout.index(self.episodes, episode.id)

    def offset(self, episode):
        if self.episode_off is None:
            raise Exception
        return self.episode_off[self.get_index_for_episode(episode)]

    def __iter__(self):
        if self.episodes is None:
            raise Exception
        return EpisodeIter(self)

    def __getitem__(self, item):
        if self.episodes is None:
            raise Exception
        index = RedisRollout.find_le(self.episode_off, item)
        step_i = item - self.episode_off[index]
        encoded_step = self.redis.lindex(self.episodes[index], step_i)
        step = self.env_config.step_coder.decode(encoded_step)
        return step

    def __len__(self):
        """
        total number of steps in rollout
        to iterate episodes, iterate over rollout object
            for episode in rollout:
                pass
        """
        steps = self.redis.get(self.key('steps'))
        if steps is not None:
            return int(steps)
        else:
            return 0


class EpisodeIter:
    def __init__(self, rollout):
        self.rollout = rollout
        self.index = 0
        self.len = len(rollout.episodes)

    def __next__(self):
        if self.index == self.len:
            raise StopIteration
        uuid = self.rollout.episodes[self.index]
        episode = Episode(self.rollout, self.rollout.redis, uuid)
        self.index += 1
        return episode


class StepIter:
    def __init__(self, rollout, episode):
        self.rollout = rollout
        self.episode = episode
        self.index = 0
        self.len = len(episode)

    def __next__(self):
        if self.index == self.len:
            raise StopIteration
        item = self.episode[self.index]
        self.index += 1
        return item


class Episode:
    def __init__(self, rollout, redis, id):
        self.rollout = rollout
        self.redis = redis
        self.id = id
        self.total_reward_key = self.id + '_total_reward'
        self.batch = 0
        self.p = None

    def end(self):
        if self.p is not None:
            self.p.execute()

        lockname = self.rollout.key('lock')
        lk = Lock(self.redis, lockname)
        logging.debug(f'getting lock {lockname}')
        lk.acquire()
        logging.debug(f'got lock {lockname}')

        if not self.redis.exists(self.rollout.key('finalized')):
            self.redis.lpush(self.rollout.key('episodes'), self.id)
            self.redis.incrby(self.rollout.key('steps'), len(self))

        lk.release()
        logging.debug(f'released lock {lockname}')

    def total_reward(self):
        try:
            total = float(self.redis.get(self.total_reward_key))
            return total
        except (ValueError, TypeError):
            logging.error('Error while getting total reward, returning 0')
            return 0

    def append(self, step, batch=1):
        encoded = self.rollout.env_config.step_coder.encode(step)
        self.batch += 1
        if self.p is None:
            self.p = self.redis.pipeline()

        self.p.rpush(self.id, encoded)
        if step.reward != 0:
            self.p.incrbyfloat(self.total_reward_key, step.reward)

        if self.batch % batch == 0:
            self.p.execute()
            self.p = None

    def __getitem__(self, item):
        encoded_step = self.redis.lindex(self.id, item)
        step = self.rollout.env_config.step_coder.decode(encoded_step)
        return step

    def __len__(self):
        return self.redis.llen(self.id)

    def __iter__(self):
        return StepIter(self.rollout, self)


class NumpyCoder:
    def __init__(self, num_axes, dtype):
        self.header_fmt = '>'
        for _ in range(num_axes):
            self.header_fmt += 'I'
        self.header_size = struct.calcsize(self.header_fmt)
        self.dtype = dtype

    def encode(self, observation):
        shape = struct.pack(self.header_fmt, *observation.shape)
        b = observation.tobytes()
        return shape + b

    def decode(self, encoded):
        shape = struct.unpack(self.header_fmt, encoded[:self.header_size])
        o = np.frombuffer(encoded, dtype=self.dtype, offset=self.header_size).reshape(shape)
        return o


class StepCoder:
    def __init__(self, observation_coder):
        self.o_coder = observation_coder
        self.header_fmt = '>if?'
        self.header_len = struct.calcsize(self.header_fmt)

    def encode(self, step):
        """ encode step to bstring"""
        b_ard = struct.pack(self.header_fmt, step.action, step.reward, step.done)
        b_o = self.o_coder.encode(step.observation)
        encoded = b_ard + b_o
        return encoded

    def decode(self, encoded):
        """ decode step from bstring"""
        action, reward, done = struct.unpack(self.header_fmt, encoded[:self.header_len])
        o = self.o_coder.decode(encoded[self.header_len:])
        return Step(o, action, reward, done)


class Step:
    def __init__(self, observation, action, reward, done):
        self.observation = observation
        self.reward = reward
        self.action = action
        self.done = done


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
