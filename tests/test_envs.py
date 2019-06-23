import gym
import tests.envs
import numpy as np
import tests.envs
import torch
from pickle import dumps, loads


def test_linewalk():
    env = gym.make('LineWalk-v0')
    env.unwrapped.length = 4

    obs = env.reset()

    assert np.allclose(obs, np.array([1.0, 0.0, 0.0, 0.0]))

    obs, reward, done, info = env.step(1)

    assert np.allclose(obs, np.array([0.0, 1.0, 0.0, 0.0]))
    assert reward == 0
    assert done == False

    obs, reward, done, info = env.step(1)

    assert np.allclose(obs, np.array([0.0, 0.0, 1.0, 0.0]))
    assert reward == 0
    assert done == False

    obs, reward, done, info = env.step(1)

    assert np.allclose(obs, np.array([0.0, 0.0, 0.0, 1.0]))
    assert reward == 1.0
    assert done == True

    obs = env.reset()

    assert np.allclose(obs, np.array([1.0, 0.0, 0.0, 0.0]))

    obs, reward, done, info = env.step(0)

    assert np.allclose(obs, np.array([1.0, 0.0, 0.0, 0.0]))
    assert reward == 0
    assert done == False

    obs, reward, done, info = env.step(1)
    obs, reward, done, info = env.step(0)

    assert np.allclose(obs, np.array([1.0, 0.0, 0.0, 0.0]))
    assert reward == 0
    assert done == False


def test_bandit():
    env = gym.make('Bandit-v0')
    obs = env.reset()
    assert np.allclose(obs, np.array([0.0, 1.0, 0.0]))

    obs, reward, done, info = env.step(0)
    assert np.allclose(obs, np.array([1.0, 0.0, 0.0]))
    assert reward == -1.0
    assert done is True

    obs = env.reset()
    obs, reward, done, info = env.step(1)
    assert np.allclose(obs, np.array([0.0, 0.0, 1.0]))
    assert reward == 1.0
    assert done is True


def test_bandit_save_restore():
    env = gym.make('Bandit-v0')
    obs = env.reset()
    assert np.allclose(obs, np.array([0.0, 1.0, 0.0]))

    prev = dumps(env)

    obs, reward, done, info = env.step(0)
    assert np.allclose(obs, np.array([1.0, 0.0, 0.0]))
    assert reward == -1.0
    assert done is True

    env = loads(prev)
    obs, reward, done, info = env.step(1)
    assert np.allclose(obs, np.array([0.0, 0.0, 1.0]))


def test_bandit_lookahead():
    env = gym.make('BanditLookahead-v0')
    obs = env.reset()
    assert np.allclose(obs, np.array([[0.0, 0.0, 1.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 1.0, 0.0]
                                      ]))

    t = tests.envs.BanditLookaheadTransform()

    transformed = t(obs, torch.float)
    assert torch.allclose(transformed, torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]))

    obs, reward, done, info = env.step(0)
    transformed = t(obs, torch.float)
    assert np.allclose(obs, np.array([[0.0, 1.0, 0.0, 0.0, 0.0],
                                      [1.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 1.0, 0.0, 0.0]
                                      ]))
    assert torch.allclose(transformed, torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]))
    assert reward == 0.0
    assert done is False

    obs = env.reset()
    obs, reward, done, info = env.step(1)
    transformed = t(obs, torch.float)

    assert np.allclose(obs, np.array([[0.0, 0.0, 0.0, 1.0, 0.0],
                                      [0.0, 0.0, 1.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 1.0]
                                      ]))
    assert torch.allclose(transformed, torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]))
    assert reward == 0.0
    assert done is False

