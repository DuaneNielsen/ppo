from data import RolloutDatasetBase, StepCoder, NumpyCoder
import configs
import data
from statistics import mean, stdev
import pytest
import numpy as np
from torch.utils.data import DataLoader

@pytest.fixture()
def db():
    db = data.Db()
    yield db
    db.drop()


def test_advantage(db):
    config = configs.BaseConfig('test_data', StepCoder(observation_coder=NumpyCoder(2, dtype=np.float64)))

    rollout = db.create_rollout(config)
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

    rollout = db.latest_rollout(config)

    dataset = data.RolloutDatasetBase(config, rollout)

    a = []
    a.append(1.0 * config.discount_factor ** 2)
    a.append(1.0 * config.discount_factor)
    a.append(1.0)
    a.append(1.0 * config.discount_factor ** 2)
    a.append(1.0 * config.discount_factor)
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

def test_concurrency(db):
    config = configs.BaseConfig('test_data', StepCoder(observation_coder=NumpyCoder(2, dtype=np.float64)))

    rollout = db.create_rollout(config)
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

    rollout = db.latest_rollout(config)

    dataset = data.RolloutDatasetBase(config, rollout)


    episode = rollout.create_episode()
    o1 = np.random.rand(80, 80)
    episode.append(data.Step(o1, 1, 0.0, False))
    o2 = np.random.rand(80, 80)
    episode.append(data.Step(o2, 1, 0.0, False))
    o3 = np.random.rand(80, 80)
    episode.append(data.Step(o3, 1, 1.0, False))
    obs.append([o1, o2, o3])
    episode.end()

    loader = DataLoader(dataset, batch_size=65, shuffle=True)

    for minibatch in loader:
        pass

