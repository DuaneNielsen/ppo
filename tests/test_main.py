import data
import numpy as np
import configs
import pytest
from statistics import mean, stdev
from data import StepCoder, NumpyCoder, RedisSequence, AdvancedNumpyCoder, AdvancedStepCoder, RewardDoneCoder, Step
import threading
import time
import random
from models import PPOWrap
from services.trainer import Trainer
from services.gatherer import Gatherer
from services.coordinator import Coordinator
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


def test_advanced_numpy():
    coder1 = data.AdvancedNumpyCoder(shape=(2, 2), dtype=np.float64)
    end = coder1.set_offset(0)
    ndarray = np.random.rand(2, 2)
    encoded = coder1.encode(ndarray)
    decoded = coder1.decode(encoded)
    np.testing.assert_array_equal(ndarray, decoded)

    coder2 = data.AdvancedNumpyCoder(shape=(2, 2), dtype=np.float64)
    end = coder2.set_offset(end)
    ndarray1 = np.random.rand(2, 2).astype(np.float64)
    ndarray2 = np.random.rand(2, 2).astype(np.float64)
    encoded = coder1.encode(ndarray1)
    encoded += coder2.encode(ndarray2)
    decoded1 = coder1.decode(encoded)
    decoded2 = coder2.decode(encoded)

    np.testing.assert_array_equal(ndarray1, decoded1)
    np.testing.assert_array_equal(ndarray2, decoded2)


def test_advanced_numpy_mixed():
    coder1 = data.AdvancedNumpyCoder(shape=(2, 2), dtype=np.float32)
    coder2 = data.AdvancedNumpyCoder(shape=(2, 2), dtype=np.float16)
    end = coder1.set_offset(0)
    end = coder2.set_offset(end)
    ndarray1 = np.random.rand(2, 2).astype(np.float32)
    ndarray2 = np.random.rand(2, 2).astype(np.float16)
    encoded = coder1.encode(ndarray1)
    encoded += coder2.encode(ndarray2)
    decoded1 = coder1.decode(encoded)
    decoded2 = coder2.decode(encoded)

    np.testing.assert_array_equal(ndarray1, decoded1)
    np.testing.assert_array_equal(ndarray2, decoded2)


def test_advanced_numpy_mixed():
    coder1 = data.AdvancedNumpyCoder(shape=(2, 2), dtype=np.float32)
    coder2 = data.AdvancedNumpyCoder(shape=4, dtype=np.float16)
    end = coder1.set_offset(0)
    end = coder2.set_offset(end)
    ndarray1 = np.random.rand(2, 2).astype(np.float32)
    ndarray2 = np.random.rand(4).astype(np.float16)
    encoded = coder1.encode(ndarray1)
    encoded += coder2.encode(ndarray2)
    decoded1 = coder1.decode(encoded)
    decoded2 = coder2.decode(encoded)

    np.testing.assert_array_equal(ndarray1, decoded1)
    np.testing.assert_array_equal(ndarray2, decoded2)


def test_reward_done_coder():
    reward = 1.4
    done = True
    coder = RewardDoneCoder()
    coder.set_offset(0)
    encoded = coder.encode(reward, done)
    reward_decoded, done_decoded = coder.decode(encoded)
    assert reward == reward_decoded
    assert done == done_decoded


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


def test_advanced_stepcoder():
    coder = AdvancedStepCoder(state_shape=(24,), state_dtype=np.float32, action_shape=(6,), action_dtype=np.float32)
    state = np.random.rand(24).astype(dtype=np.float32)
    action = np.random.rand(6).astype(dtype=np.float32)
    reward = 4.5
    done = True
    step = Step(state, action, reward, done)
    encoded = coder.encode(step)
    step_d = coder.decode(encoded)
    np.testing.assert_array_equal(step_d.observation, step.observation)
    np.testing.assert_array_equal(step_d.action, step.action)
    assert step.reward == step_d.reward
    assert step.done == step_d.done


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
