import torch
from models import *
from math import *
import numpy as np

def test_qf():
    q_f = QMLP(features=10, actions=4, hidden=10)
    target = torch.ones(5)
    state = torch.rand(5, 10)
    action = torch.rand(5, 4)
    optim = torch.optim.SGD(q_f.parameters(), lr=0.02)

    for _ in range(500):
        optim.zero_grad()
        value = q_f(state, action)

        loss = torch.mean((target - value) ** 2)
        loss.backward()
        optim.step()

    value = q_f(state, action)

    assert torch.allclose(target, value, atol=1e-1)


def test_greedy_dist():
    p = torch.tensor([[0.2867, 0.1678, 0.2680, 0.1393, 0.1382],
                      [0.1759, 0.2696, 0.1191, 0.2278, 0.2076]])
    gd = GreedyDiscreteOneHotDist(p)
    a = gd.sample()
    logprob = gd.logprob(a)

    assert a[0, 0].item() == 1
    assert a[1, 1].item() == 1
    assert logprob[0].item() - log(0.2867) < 1e-5
    assert logprob[1].item() - log(0.2696) < 1e-5


def test_epsilon_greedy_dist():
    p = torch.tensor([[0.5, 0.125, 0.125, 0.25]])
    gd = EpsilonGreedyDiscreteDist(p, epsilon=0.05)

    samples = torch.zeros_like(p)

    for _ in range(10000):
        a = gd.sample()
        index = torch.argmax(a)
        samples[0, index] += 1.0

    p_sampled = samples / 10000
    assert torch.allclose(torch.tensor([1.0 - 0.05, 0.05 / 3, 0.05 / 3, 0.05 / 3]), p_sampled, atol=1e-2)

    prob = gd.logprob(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert gd.logprob(torch.tensor([[1.0, 0.0, 0.0, 0.0]])).item() - log(1.0 - 0.05) < 1e-5
    assert gd.logprob(torch.tensor([[0.0, 1.0, 0.0, 0.0]])).item() - log(0.05 / 3) < 1e-5


class TestQF(nn.Module):
    def __init__(self, actions):
        super().__init__()
        self.actions = actions

    def forward(self, state, action):
        value = torch.sum(state * action, dim=1)
        return value


def test_greedy_policy():
    state = torch.rand(2, 3)
    q_f = QMLP(features=3, actions=4, hidden=10)
    policy = ValuePolicy(q_f, GreedyDiscreteOneHotDist)
    a_dist = policy(state)
    a_dist = a_dist.sample()
    assert True


def test_eps_greedy_policy():
    state = torch.rand(1, 3)
    q_f = QMLP(features=3, actions=4, hidden=10)
    policy = ValuePolicy(q_f, EpsilonGreedyDiscreteDist, epsilon=0.10)
    a_dist = policy(state)
    a_dist = a_dist.sample()
    assert True


def test_greedy_with_dummy():
    state = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    qf = TestQF(actions=4)
    policy = ValuePolicy(qf, GreedyDiscreteOneHotDist)
    a_dist = policy(state)

    # todo make unit test that works with softmax

    # assert a_dist.probs[0, 0] == 0.1
    # assert a_dist.probs[0, 1] == 0.2
    # assert a_dist.probs[0, 2] == 0.3
    # assert a_dist.probs[0, 3] == 0.4
    #
    # assert a_dist.probs[1, 0] == 0.4
    # assert a_dist.probs[1, 1] == 0.3
    # assert a_dist.probs[1, 2] == 0.2
    # assert a_dist.probs[1, 3] == 0.1


def test_qtable():
    qtable = QTable(3, 2)
    state = torch.tensor([[0.0, 0.0, 1.0],
                          [1.0, 0.0, 0.0]])

    action = torch.tensor([[0.0, 1.0],
                           [1.0, 0.0]])

    target = torch.tensor([-1.0, 1.0])
    optim = torch.optim.SGD(qtable.parameters(), lr=0.1)
    for _ in range(10):
        optim.zero_grad()
        value = qtable(state, action)
        loss = torch.sum((target - value) ** 2)
        loss.backward()
        optim.step()

    print(qtable.weights)


def test_value_lookuptable():
    table = TestValueFunction(3)
    state = torch.tensor([[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0],
                          [0.0, 0.0, 1.0]
                          ])
    target = torch.tensor([-1.0, 0.0, 1.0, 1.0])

    optim = torch.optim.SGD(table.parameters(), lr=0.05)

    for _ in range(20):
        optim.zero_grad()
        value = table(state)
        loss = torch.sum((target - value) ** 2)
        loss.backward()
        optim.step()

    print(table.weights)
    assert torch.allclose(table.weights, target[:3], atol=0.2)


def test_policy_lookuptable():
    table = LookupDiscreteTestPolicy(3, 2)
    states = torch.tensor([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0],
                           [0.0, 0.0, 1.0]
                           ])

    targets = torch.tensor([[0.0, 1.0],
                            [1.0, 0.0],
                            [0.0, 1.0],
                            [0.0, 1.0],
                            ])

    optim = torch.optim.SGD(table.parameters(), lr=0.1)

    for _ in range(50):
        optim.zero_grad()
        probs = table(states).probs
        loss = torch.sum((targets - probs) ** 2)
        loss.backward()
        optim.step()

    expected_weights = torch.tensor([[0.0, 1.0],
                                     [1.0, 0.0],
                                     [0.0, 1.0],
                                     ])

    assert torch.allclose(table.probs, expected_weights, atol=0.2)


def test_preconfigured_value_lookup():

    table = TestValueFunction(3)
    state = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    table.weights = nn.Parameter(torch.tensor([-1.0, 0.0, 1.0]))

    v0 = table(state[0].unsqueeze(0))
    assert v0.item() == -1.0
    v1 = table(state[1].unsqueeze(0))
    assert v1.item() == 0.0
    v2 = table(state[2].unsqueeze(0))
    assert v2.item() == 1.0


def test_best_value_policy():
    state = torch.tensor([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    batch = torch.stack((state, state), dim=0)

    vf = TestValueFunction(3)
    vf.weights = nn.Parameter(torch.tensor([-1.0, 0.0, 1.0]))

    policy = BestValuePolicy(vf)

    dist = policy(batch)
    print(dist.probs)
    print(dist.sample())