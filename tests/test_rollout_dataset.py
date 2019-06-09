from data import *
import configs
import data
from statistics import mean, stdev
import pytest
import numpy as np
from torch.utils.data import DataLoader
from models import OneHotDiscreteActionTransform
import torch


@pytest.fixture()
def db():
    db = data.Db('localhost', 6379, None, 1)
    try:
        yield db
    finally:
        db.drop()


def test_advantage(db):
    config = configs.DiscreteConfig(gym_env_string='test',
                                    features=(80, 80),
                                    features_dtype=np.float64,
                                    action_map=[0, 1, 6])
    config.discount_factor = 0.99

    rollout = db.create_rollout(config)
    obs = []

    episode = rollout.create_episode()
    o1 = np.random.rand(80, 80)
    a = config.default_action
    episode.append(data.Step(o1, a, 0.0, False))
    o2 = np.random.rand(80, 80)
    episode.append(data.Step(o2, 6, 0.0, False))
    o3 = np.random.rand(80, 80)
    episode.append(data.Step(o3, 1, 1.0, False))
    # sentinel containing last observation
    episode.append(data.Step(o3, a, 0.0, True))
    obs.append([o1, o2, o3])
    episode.end()

    rollout_id = rollout.id

    rollout = db.latest_rollout(config)

    assert rollout.id == rollout_id

    dataset = data.SARAdvantageDataset(rollout, discount_factor=0.99, state_transform=config.transform,
                                       action_transform=config.action_transform)

    a = []
    a.append(1.0 * config.discount_factor ** 2)
    a.append(1.0 * config.discount_factor)
    a.append(1.0)

    mu = mean(a)
    sigma = stdev(a)

    adv = [(vl - mu) / (sigma + 1e-12) for vl in a]

    assert len(dataset) == 3

    observation, action, reward, advantage = dataset[0]
    assert reward == 0.00
    assert advantage == adv[0]
    assert action == config.default_action

    observation, action, reward, advantage = dataset[1]
    assert reward == 0.0
    assert advantage == adv[1]
    assert action.item() == 2

    observation, action, reward, advantage = dataset[2]
    assert reward == 1.0
    assert advantage == adv[2]
    assert action.item() == 1


def test_concurrency(db):
    config = configs.DiscreteConfig(gym_env_string='test',
                                    features=(80, 80),
                                    features_dtype=np.float64,
                                    action_map=[0, 1])

    rollout = db.create_rollout(config)
    obs = []

    episode = rollout.create_episode()
    o1 = np.random.rand(80, 80)
    a = config.default_action
    episode.append(data.Step(o1, a, 0.0, False))
    o2 = np.random.rand(80, 80)
    episode.append(data.Step(o2, a, 0.0, False))
    o3 = np.random.rand(80, 80)
    episode.append(data.Step(o3, a, 1.0, False))
    # sentinel containing last observation
    episode.append(data.Step(o3, a, 0.0, True))
    obs.append([o1, o2, o3])
    episode.end()

    episode = rollout.create_episode()
    o1 = np.random.rand(80, 80)
    episode.append(data.Step(o1, a, 0.0, False))
    o2 = np.random.rand(80, 80)
    episode.append(data.Step(o2, a, 0.0, False))
    o3 = np.random.rand(80, 80)
    episode.append(data.Step(o3, a, 1.0, False))
    # sentinel containing last observation
    episode.append(data.Step(o3, a, 0.0, True))
    obs.append([o1, o2, o3])
    episode.end()


    rollout = db.latest_rollout(config)

    dataset = data.SARAdvantageDataset(rollout, discount_factor=0.99, state_transform=config.transform,
                                       action_transform=config.action_transform)

    episode = rollout.create_episode()
    o1 = np.random.rand(80, 80)
    episode.append(data.Step(o1, a, 0.0, False))
    o2 = np.random.rand(80, 80)
    episode.append(data.Step(o2, a, 0.0, False))
    o3 = np.random.rand(80, 80)
    episode.append(data.Step(o3, a, 1.0, False))
    # sentinel containing last observation
    episode.append(data.Step(o3, a, 0.0, True))
    obs.append([o1, o2, o3])
    episode.end()


    loader = DataLoader(dataset, batch_size=65, shuffle=True)

    for minibatch in loader:
        pass


def add_episode(exp_buffer):

    episode = exp_buffer.create_episode()
    step1 = data.Step(np.random.rand(80, 80), 1, 0.0, False)
    step2 = data.Step(np.random.rand(80, 80), 0, 0.0, False)
    terminal = data.Step(np.random.rand(80, 80), 1, 0.0, True)
    episode.append(step1)
    episode.append(step2)
    episode.append(terminal)
    episode.end()
    return step1, step2, terminal


def test_SARS_dataset(db):
    config = configs.DiscreteConfig(gym_env_string='test',
                                    features=(80, 80),
                                    features_dtype=np.float64,
                                    action_map=[0, 1])

    exp_buffer = db.create_rollout(config)
    step1, step2, step3 = add_episode(exp_buffer)
    action_transform = OneHotDiscreteActionTransform(config.action_map)

    dataset = SARSDataset(exp_buffer, state_transform=config.transform, action_transform=action_transform, precision=torch.float32)

    assert len(dataset) == 2

    state, action, reward, next_state = dataset[0]
    assert torch.allclose(action, torch.tensor([0, 1], dtype=torch.float32))
    state, action, reward, next_state = dataset[1]
    assert torch.allclose(action, torch.tensor([1, 0], dtype=torch.float32))


def add_dud_episode(exp_buffer):
    episode = exp_buffer.create_episode()
    o1 = np.random.rand(80, 80)
    episode.append(data.Step(o1, 0, 0.0, True))
    episode.end()


def test_SARS_dataset_edge_1(db):
    config = configs.DiscreteConfig(gym_env_string='test',
                                    features=(80, 80),
                                    features_dtype=np.float64,
                                    action_map=[0, 1])

    exp_buffer = db.create_rollout(config)

    add_dud_episode(exp_buffer)
    action_transform = OneHotDiscreteActionTransform(config.action_map)
    dataset = SARSDataset(exp_buffer, state_transform=config.transform, action_transform=action_transform, precision=torch.float32)

    assert len(dataset) == 0


def test_SARS_dataset_2episodes(db):
    config = configs.DiscreteConfig(gym_env_string='test',
                                    features=(80, 80),
                                    features_dtype=np.float64,
                                    action_map=[0, 1])

    exp_buffer = db.create_rollout(config)
    add_episode(exp_buffer)
    add_episode(exp_buffer)

    action_transform = OneHotDiscreteActionTransform(config.action_map)
    dataset = SARSDataset(exp_buffer, state_transform=config.transform, action_transform=action_transform, precision=torch.float32)

    assert len(dataset) == 4

    state, action, reward, next_state = dataset[0]
    state, action, reward, next_state = dataset[1]
    state, action, reward, next_state = dataset[2]
    state, action, reward, next_state = dataset[3]



def test_SARS_dataset_edge_2(db):
    config = configs.DiscreteConfig(gym_env_string='test',
                                    features=(80, 80),
                                    features_dtype=np.float64,
                                    action_map=[0, 1])

    exp_buffer = db.create_rollout(config)
    add_dud_episode(exp_buffer)
    add_episode(exp_buffer)
    add_dud_episode(exp_buffer)

    action_transform = OneHotDiscreteActionTransform(config.action_map)
    dataset = SARSDataset(exp_buffer, state_transform=config.transform, action_transform=action_transform, precision=torch.float32)

    assert len(dataset) == 2

    state, action, reward, next_state = dataset[0]
    state, action, reward, next_state = dataset[1]
