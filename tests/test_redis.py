import data
import numpy as np
import configs
import pytest
from statistics import mean, stdev
from data import StepCoder, NumpyCoder, RedisSequence


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
