from models import *
import torch
import numpy as np


def test_default_transform():
    observation = np.random.rand(10, 10)
    transform = DefaultTransform()
    observation_t = transform(observation)
    assert torch.allclose(observation_t, torch.from_numpy(observation.astype(np.float32)))


def test_discrete_action_trasform():
    action_map = [0, 1, 6]
    transform = DiscreteActionTransform(action_map)
    assert transform(torch.tensor([[2]])) == 6
    assert transform.invert(6) == torch.tensor([2])


def test_one_hot_action_transform():
    action_map = [0, 1, 6]
    transform = OneHotDiscreteActionTransform(action_map)
    assert transform(torch.tensor([[0.0, 0.0, 1.0]])) == 6
    assert torch.allclose(transform.invert(6, torch.float32),  torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32))
    assert transform(torch.tensor([[0.0, 1.0, 0.1]])) == 1
    assert torch.allclose(transform.invert(1, torch.float32),  torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32))
