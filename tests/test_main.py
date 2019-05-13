import data
import numpy as np
import configs
import pytest
from statistics import mean, stdev
from data import StepCoder, NumpyCoder, RedisSequence
import threading
import time
import random
from models import PPOWrap
from controller import Gatherer, Trainer, Coordinator
from messages import RolloutMessage
import redis

class GatherThread(threading.Thread):
    def run(self):
        s = Gatherer()
        s.main()


class TrainerThread(threading.Thread):
    def run(self):
        s = Trainer()
        s.main()


def test_buffered():
    obs = np.array([0.0, 1.0, 2.0])
    action = 1
    reward = 0.0
    s = data.Step(obs, action, reward, False)
    r = data.Step(obs, action, 1.0, True)

    episode = [s, s, s, s, s, s, s, r]
    bds = data.BufferedRolloutDataset(discount_factor=0.99)

    for step in episode:
        bds.append(step.observation, step.action, step.reward, step.done)

    # for step in bds.rollouts[0]:
    #     print(step.advantage)

    assert abs(bds.rollouts[0].advantage - 0.932) <= 0.01


def populate():
    obs = np.array([0.0, 1.0, 2.0])
    action = 1
    reward = 0.0
    s = data.Step(obs, action, reward, False)
    r = data.Step(obs, action, 1.0, True)
    episode = [s, s, s, s, s, s, s, r]
    bds = data.BufferedRolloutDataset(discount_factor=0.99)
    for step in episode:
        bds.append(step.observation, step.action, step.reward, step.done, episode='player1')
        bds.append(step.observation, step.action, step.reward, step.done, episode='player2')
    return bds


def test_double():
    bds = populate()

    # for step in bds.rollouts[0]:
    #     print(step.advantage)

    assert abs(bds.rollouts[0].advantage - 0.932) <= 0.01
    assert abs(bds.rollouts[1 + 8].advantage - 0.932) <= 0.01


def test_end_rollout():
    bds = populate()

    assert abs(bds.rollouts[0].advantage - 0.932) <= 0.01

    bds.end_rollout()

    assert abs(bds[15][3]) - 1.493 < 0.001


def test_total_reward():
    bds = populate()
    assert bds.total_reward() == 2.0


def test_len_get():
    bds = populate()
    assert len(bds) == 16
    assert bds[15][1] == 1.0
    assert bds[7][1] == 1.0
    assert bds[6][1] == 0.0
    assert type(bds[0][0]).__name__ == 'Tensor'


def test_encode_numpy():
    coder = data.NumpyCoder(1, np.float64)
    assert coder.header_fmt == '>I'

    o = np.random.rand(80)
    encoded = coder.encode(o)
    decoded = coder.decode(encoded)

    np.testing.assert_array_equal(o, decoded)

    coder = data.NumpyCoder(2, np.float64)
    assert coder.header_fmt == '>II'

    o = np.random.rand(80, 80)
    encoded = coder.encode(o)
    decoded = coder.decode(encoded)

    np.testing.assert_array_equal(o, decoded)


def test_encode_decode():
    o = np.random.rand(80, 80)
    s = data.Step(o, 1, 1.0, False)
    coder = data.StepCoder(observation_coder=data.NumpyCoder(2, np.float64))
    encoded = coder.encode(s)
    decoded = coder.decode(encoded)
    assert s.action == decoded.action
    assert s.reward == decoded.reward
    assert s.done == decoded.done
    np.testing.assert_array_equal(s.observation, decoded.observation)


def testMultiProcessRedisSquence(db):

    for _ in range(10):
        barrier = threading.Barrier(3)
        ids = []

        class TestThread(threading.Thread):
            def __init__(self):
                super().__init__()
                self.seq = RedisSequence(db.redis, 'thread_test')

            def run(self):
                ids.append(next(self.seq))
                time.sleep(random.random()/5.0)
                ids.append(next(self.seq))
                time.sleep(random.random()/5.0)
                ids.append(next(self.seq))
                barrier.wait()

        t1 = TestThread()
        t2 = TestThread()
        t1.start()
        t2.start()
        barrier.wait()

        for i, item in enumerate(ids):
            ids2 = ids.copy()
            del ids2[i]
            for j in ids2:
                assert j != item


def testGatherers():
    env_config = configs.LunarLander()
    policy_net = PPOWrap(env_config.features, env_config.action_map, env_config.hidden)

    t1 = TrainerThread()
    g1 = GatherThread()

    t1.start()
    g1.start()

    r = redis.Redis()

    RolloutMessage(0, policy_net, env_config).send(r)

    time.sleep(10)

    t1.join()
    g1.join()