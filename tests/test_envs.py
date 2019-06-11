import gym
import tests.envs
import numpy as np


def test_linewalk():
    env = gym.make('LineWalk-v0')

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
    assert reward == 0
    assert done is True

    obs = env.reset()
    obs, reward, done, info = env.step(1)
    assert np.allclose(obs, np.array([0.0, 0.0, 1.0]))
    assert reward == 1.0
    assert done is True
