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
import policy_db
import torch
import uuid


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


@pytest.fixture()
def db():
    db = data.Db()
    yield db
    db.drop()


def test_redis_write_step(db):
    env_config = configs.BaseConfig('test', data.StepCoder(data.NumpyCoder(num_axes=2, dtype=np.float)))
    rollout = db.create_rollout(env_config)
    lookup = {}

    episode = rollout.create_episode()
    o1 = np.random.rand(80, 80)
    episode.append(data.Step(o1, 1, 1.0, False))
    episode.append(data.Step(o1, 1, 1.0, False))
    episode.append(data.Step(o1, 1, 1.0, False))
    episode.end()

    episode = rollout.create_episode()
    o2 = np.random.rand(80, 80)
    episode.append(data.Step(o1, 1, 1.0, False))
    episode.append(data.Step(o1, 1, 1.0, False))
    episode.append(data.Step(o1, 1, 1.0, False))
    episode.end()

    episode = rollout.create_episode()
    o3 = np.random.rand(80, 80)
    episode.append(data.Step(o1, 1, 1.0, False))
    episode.append(data.Step(o1, 1, 1.0, False))
    episode.append(data.Step(o1, 1, 1.0, False))
    episode.end()

    rollout.end()

    step = rollout[0]
    step = rollout[1]
    step = rollout[2]

    np.testing.assert_array_equal(step.observation, o1)

    step = rollout[3]
    step = rollout[4]
    step = rollout[5]

    np.testing.assert_array_equal(step.observation, o1)

    step = rollout[6]
    step = rollout[7]
    step = rollout[8]

    np.testing.assert_array_equal(step.observation, o1)

    assert len(rollout) == 9
    assert step.reward == 1.0
    assert step.action == 1


def test_epi_batching(db):
    env_config = configs.BaseConfig('test', data.StepCoder(data.NumpyCoder(num_axes=2, dtype=np.float)))
    rollout = db.create_rollout(env_config)
    episode = rollout.create_episode()

    for _ in range(3):

        o1 = np.random.rand(80, 80)
        episode.append(data.Step(o1, 1, 1.0, False), batch=10)
        o2 = np.random.rand(80, 80)
        episode.append(data.Step(o2, 1, 1.0, False), batch=10)
        o3 = np.random.rand(80, 80)
        episode.append(data.Step(o3, 1, 1.0, False), batch=10)

    assert len(episode) == 0
    episode.append(data.Step(o3, 1, 1.0, False), batch=10)
    assert len(episode) == 10

    episode = rollout.create_episode()
    for _ in range(3):

        o1 = np.random.rand(80, 80)
        episode.append(data.Step(o1, 1, 1.0, False), batch=10)
        o2 = np.random.rand(80, 80)
        episode.append(data.Step(o2, 1, 1.0, False), batch=10)
        o3 = np.random.rand(80, 80)
        episode.append(data.Step(o3, 1, 1.0, False), batch=10)

    episode.end()

    assert len(episode) == 9


def test_epi_total_reward(db):
    env_config = configs.BaseConfig('test', data.StepCoder(data.NumpyCoder(num_axes=2, dtype=np.float)))
    rollout = db.create_rollout(env_config)
    episode = rollout.create_episode()

    assert episode.total_reward() == 0

    o1 = np.random.rand(80, 80)
    episode.append(data.Step(o1, 1, 1.0, False))

    assert episode.total_reward() == 1.0

    o2 = np.random.rand(80, 80)
    episode.append(data.Step(o2, 1, 1.0, False))

    assert episode.total_reward() == 2.0

    o3 = np.random.rand(80, 80)
    episode.append(data.Step(o3, 1, 1.0, False))

    assert episode.total_reward() == 3.0




def test_step_iter(db):
    env_config = configs.BaseConfig('test', data.StepCoder(data.NumpyCoder(num_axes=2, dtype=np.float)))
    rollout = db.create_rollout(env_config)

    episode = rollout.create_episode()
    o1 = np.random.rand(80, 80)
    episode.append(data.Step(o1, 1, 1.0, False))
    o2 = np.random.rand(80, 80)
    episode.append(data.Step(o2, 1, 1.0, False))
    o3 = np.random.rand(80, 80)
    episode.append(data.Step(o3, 1, 1.0, False))
    episode.end()

    assert len(episode) == 3

    for step, o in zip(episode, [o1, o2, o3]):
        np.testing.assert_array_equal(step.observation, o)


def test_epi_iter(db):
    env_config = configs.BaseConfig('test', StepCoder(observation_coder=NumpyCoder(2, dtype=np.float64)))
    rollout = db.create_rollout(env_config)

    lookup = {}

    episode = rollout.create_episode()
    o1 = np.random.rand(80, 80)
    episode.append(data.Step(o1, 1, 1.0, False))
    o2 = np.random.rand(80, 80)
    episode.append(data.Step(o2, 1, 1.0, False))
    o3 = np.random.rand(80, 80)
    episode.append(data.Step(o3, 1, 1.0, False))
    lookup[episode.id] = [o1, o2, o3]
    episode.end()

    episode = rollout.create_episode()
    o4 = np.random.rand(80, 80)
    episode.append(data.Step(o4, 1, 1.0, False))
    o5 = np.random.rand(80, 80)
    episode.append(data.Step(o5, 1, 1.0, False))
    o6 = np.random.rand(80, 80)
    episode.append(data.Step(o6, 1, 1.0, False))
    lookup[episode.id] = [o4, o5, o6]
    episode.end()

    episode = rollout.create_episode()
    o7 = np.random.rand(80, 80)
    episode.append(data.Step(o7, 1, 1.0, False))
    o8 = np.random.rand(80, 80)
    episode.append(data.Step(o8, 1, 1.0, False))
    o9 = np.random.rand(80, 80)
    episode.append(data.Step(o9, 1, 1.0, False))
    lookup[episode.id] = [o7, o8, o9]
    episode.end()

    rollout.end()

    assert len(rollout) == 9

    for episode in rollout:
        assert len(episode) == 3
        ob = lookup[episode.id]
        for step, o in zip(episode, ob):
            np.testing.assert_array_equal(step.observation, o)



def test_advantage(db):
    env_config = configs.BaseConfig('test_data', StepCoder(observation_coder=NumpyCoder(2, dtype=np.float64)))

    rollout = db.create_rollout(env_config)
    obs = []

    episode = rollout.create_episode()
    o1 = np.random.rand(80, 80)
    episode.append(data.Step(o1, 1, 0.0, False))
    o2 = np.random.rand(80, 80)
    episode.append(data.Step(o2, 1, 0.0, False))
    o3 = np.random.rand(80, 80)
    episode.append(data.Step(o3, 1, 1.0, False))
    obs.append([o1, o2, o3])
    episode.end()

    episode = rollout.create_episode()
    o1 = np.random.rand(80, 80)
    episode.append(data.Step(o1, 1, 0.0, False))
    o2 = np.random.rand(80, 80)
    episode.append(data.Step(o2, 1, 0.0, False))
    o3 = np.random.rand(80, 80)
    episode.append(data.Step(o3, 1, 1.0, False))
    obs.append([o1, o2, o3])
    episode.end()

    rollout.end()

    dataset = data.RolloutDatasetBase(env_config, rollout)

    a = []
    a.append(1.0 * env_config.discount_factor ** 2)
    a.append(1.0 * env_config.discount_factor)
    a.append(1.0)
    a.append(1.0 * env_config.discount_factor ** 2)
    a.append(1.0 * env_config.discount_factor)
    a.append(1.0)

    mu = mean(a)
    sigma = stdev(a)

    adv = [(vl - mu) / (sigma + 1e-12) for vl in a]

    observation, action, reward, advantage = dataset[0]
    assert reward == 0.00
    assert advantage == adv[0]

    observation, action, reward, advantage = dataset[1]
    assert reward == 0.0
    assert advantage == adv[1]

    observation, action, reward, advantage = dataset[2]
    assert reward == 1.0
    assert advantage == adv[2]

    observation, action, reward, advantage = dataset[3]
    assert reward == 0.00
    assert advantage == adv[3]

    observation, action, reward, advantage = dataset[4]
    assert reward == 0.0
    assert advantage == adv[4]

    observation, action, reward, advantage = dataset[5]
    assert reward == 1.0
    assert advantage == adv[5]


def testRedisSequence(db):

    seq = RedisSequence(db.redis, 'test')
    assert next(seq) == 0
    assert next(seq) == 1
    assert next(seq) == 2
    assert next(seq) == 3
    seq = RedisSequence(db.redis, 'test')
    assert next(seq) == 4
    assert next(seq) == 5
    seq = RedisSequence(db.redis, 'test', reset=True)
    assert next(seq) == 0
    assert next(seq) == 1


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

def testPostgresWrite():

    config = configs.LunarLander()
    s0 = torch.randn(config.features)
    s1 = torch.randn(config.features)
    s = config.prepro(s0, s1).unsqueeze(0)

    policy_net = PPOWrap(config.features, config.action_map, config.hidden)

    a0 = policy_net(s)

    policy_state_dict = policy_net.state_dict()
    policy_net.load_state_dict(policy_state_dict)

    # co = Coordinator(redis_host='localhost', redis_port=6379, redis_db=0, redis_password=None,
    #                  db_host='localhost', db_port=5432, db_name='testpython', db_user='ppo', db_password='password')
    db = policy_db.PolicyDB()
    run = "run_id"
    run_state = 'RUNNING'
    stats = {'header': 'unittest', 'ave_reward_episode': 15.3 + random.random()}
    db.write_policy(run, run_state, policy_state_dict, stats, config)
    record = db.get_latest(run)
    policy_net.load_state_dict(record.policy)

    a1 = policy_net(s)

    assert record.run == run
    assert record.stats == stats
    assert torch.equal(a0, a1)
    assert record.config['gym_env_string'] == config.gym_env_string
    assert record.config_b.gym_env_string == config.gym_env_string


def write_policy(run, reward):
    config = configs.LunarLander()
    policy_net = PPOWrap(config.features, config.action_map, config.hidden)
    policy_state_dict = policy_net.state_dict()
    db = policy_db.PolicyDB()
    run_state = 'RUNNING'
    stats = {'header': 'unittest', 'ave_reward_episode': reward}
    db.write_policy(run, run_state, policy_state_dict, stats, config)
    return db

def testDelete():
    db = write_policy('delete_me', 200.0)

    assert db.count('delete_me') > 0

    db.delete('delete_me')

    assert db.count('delete_me') == 0

def testLatest():
    write_policy('run1', 10.0 + random.random())
    db = write_policy('run2', 10.0 + random.random())
    record = db.get_latest()

    assert record.run == 'run2'

    record = db.get_latest('run1')

    assert record.run == 'run1'

def testGetBest():
    db = write_policy('best_run', 100000.0)
    record = db.get_best('LunarLander-v2')

    assert record.stats['ave_reward_episode'] == 100000.0

def testCleanUp():
    db = policy_db.PolicyDB()
    db.delete('run_id')
    db.delete('run1')
    db.delete('run2')
    db.delete('best_run')
    db.delete('delete_me')