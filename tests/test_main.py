import unittest
import data
import numpy as np


class TestData(unittest.TestCase):
    def test_buffered(self):
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

    def populate(self):
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

    def test_end_rollout(self):
        bds = self.populate()

        assert abs(bds.rollouts[0].advantage - 0.932) <= 0.01

        bds.end_rollout()

        assert abs(bds[15][3]) - 1.493 < 0.001

    def test_total_reward(self):
        bds = self.populate()
        assert bds.total_reward() == 2.0

    def test_len_get(self):
        bds = self.populate()
        assert len(bds) == 16
        assert bds[15][1] == 1.0
        assert bds[7][1] == 1.0
        assert bds[6][1] == 0.0
        assert type(bds[0][0]).__name__ == 'Tensor'
