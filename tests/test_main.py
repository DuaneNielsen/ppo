import data
import numpy as np
import pytest


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


def test_double(self):
    bds = self.populate()

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


def test_encode_decode():
    o = np.random.rand(80, 80)
    s = data.Step(o, 1, 1.0, False)
    encoded = s.encode()
    decoded = data.Step.decode(encoded, np.float64)
    assert s.action == decoded.action
    assert s.reward == decoded.reward
    assert s.done == decoded.done
    np.testing.assert_array_equal(s.observation, decoded.observation)


@pytest.fixture()
def resource():
    yield "resource"
    db = data.Db()
    db.drop()


def test_redis_write_step(resource):
    db = data.Db()
    rollout = db.create_rollout(np.float64)

    episode = rollout.create_episode()
    o1 = np.random.rand(80, 80)
    episode.append(data.Step(o1, 1, 1.0, False))
    episode.append(data.Step(o1, 1, 1.0, False))
    episode.append(data.Step(o1, 1, 1.0, False))

    episode = rollout.create_episode()
    o2 = np.random.rand(80, 80)
    episode.append(data.Step(o2, 1, 1.0, False))
    episode.append(data.Step(o2, 1, 1.0, False))
    episode.append(data.Step(o2, 1, 1.0, False))

    episode = rollout.create_episode()
    o3 = np.random.rand(80, 80)
    episode.append(data.Step(o3, 1, 1.0, False))
    episode.append(data.Step(o3, 1, 1.0, False))
    episode.append(data.Step(o3, 1, 1.0, False))

    rollout.end()
    step = rollout[0]
    step = rollout[1]
    step = rollout[2]

    np.testing.assert_array_equal(step.observation, o1)

    step = rollout[3]
    step = rollout[4]
    step = rollout[5]

    np.testing.assert_array_equal(step.observation, o2)

    step = rollout[6]
    step = rollout[7]
    step = rollout[8]

    np.testing.assert_array_equal(step.observation, o3)

