import gym
from data.transforms import *
import gym_duane
import numpy as np
from statistics import mean
from torch import nn
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader


class BatchStep:
    def __init__(self, state, action, reward, done, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done
        self.next_state = next_state

    def __getitem__(self, item):
        return Step(self.state[item], self.action[item], self.reward[item], self.done[item], self.next_state[item])


class Step:
    def __init__(self, state, action, reward, done, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done
        self.next_state = next_state


def batch_episode(env, policy, max_rollout_len=4000, v=None, render=False, display_observation=False):
    episode = []
    entropy = []

    state = env.reset()

    if render:
        env.render()

    for _ in range(max_rollout_len):

        action_dist = policy(state)

        entropy.append(action_dist.entropy().mean().item())

        action = action_dist.sample()

        next_state, reward, done, info = env.step(action)

        episode.append(BatchStep(state.cpu().numpy(), action.cpu().numpy(), reward.cpu().numpy(), done.cpu().numpy(),
                            next_state.cpu().numpy()))
        if render:
            env.render(mode='human')
        if display_observation:
            v.render(state)

        state = next_state

    final_entropy = mean(entropy)
    return episode, final_entropy


class RandomPolicy:
    def __call__(self, state):
        p = torch.ones(state.size(0), 4) / 4
        return Categorical(p)


class EpsilonGreedyProperDiscreteDist:
    def __init__(self, probs, epsilon=0.05):
        self.probs = probs
        self.epsilon = epsilon

        e = self.epsilon / (self.probs.size(1) - 1)
        max = torch.argmax(self.probs, dim=1)
        self.p = torch.ones_like(self.probs) * e
        self.p[torch.arange(self.p.size(0)), max] = 1.0 - self.epsilon

    def sample(self):
        return Categorical(self.p).sample()

    def entropy(self):
        return torch.sum(- self.probs * torch.log2(self.probs), dim=1)

    def logprob(self, action):
        probs = torch.sum(self.p * action, dim=1)
        return torch.log(probs)


class QPolicy(nn.Module):
    def __init__(self, qf, actions, dist_class, **kwargs):
        super().__init__()
        self.qf = qf
        self.actions = torch.arange(actions)
        self.num_actions = actions
        self.dist_class = dist_class
        self.kwargs = kwargs

    def parameters(self, recurse=True):
        return self.qf.parameters(recurse)

    def forward(self, state):
        batch_size = state.size(0)
        input_size = state.shape[1:]

        # copy the states * number of actions
        states = state.unsqueeze(1).expand(batch_size, self.num_actions, *input_size)
        states = states.reshape(batch_size * self.num_actions, *input_size)

        # repeat the actions for each state in the batch
        actions = self.actions.unsqueeze(0).expand(batch_size, -1)
        actions = actions.reshape(batch_size * self.num_actions)

        values = self.qf(states, actions)
        values = values.reshape(batch_size, self.num_actions)

        probs = torch.softmax(values, dim=1)

        return self.dist_class(probs, **self.kwargs)


class DiscreteQTable(nn.Module):
    def __init__(self, features, actions):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(actions, *features))

    def forward(self, state, action):
        return torch.sum(self.weights[action, :, :] * state, dim=[1, 2])


class SARSGridDataset(Dataset):
    def __init__(self, episode):
        super().__init__()
        self.episode = episode
        self.batch_size = episode[0].state.shape[0]

    def _transform(self, step, resetting):
        state = torch.from_numpy(step.state)
        action = torch.tensor([step.action])
        reward = torch.tensor([step.reward])
        done = torch.tensor(step.done, dtype=torch.uint8)
        resetting = torch.tensor(step.done, dtype=torch.uint8)
        next_state = torch.from_numpy(step.next_state)
        return state, action, reward, done, resetting, next_state

    def __getitem__(self, item):
        t = item // self.batch_size
        offset = item % self.batch_size
        step = self.episode[t][offset]
        if t > 0:
            resetting = self.episode[t - 1][offset].done
        else:
            resetting = 0
        return self._transform(step, resetting)

    def __len__(self):
        return len(self.episode) * self.batch_size


def test_SARSGridDataset():
    state = np.array([
        [[1, 0, 0]],
        [[1, 0, 0]]
    ])
    action = np.array([
        [0],
        [0]
    ])
    reward = np.array([
        [0.0],
        [0.0]
    ])
    done = np.array([
        [1],
        [1]
    ])
    next_state = np.array([
        [[0, 1, 0]],
        [[0, 1, 0]]
    ])

    a = BatchStep(state, action, reward, done, next_state)

    state = np.array([
        [[0, 1, 0]],
        [[0, 1, 0]]
    ])
    action = np.array([
        [0],
        [0]
    ])
    reward = np.array([
        [0.0],
        [0.0]
    ])
    done = np.array([
        [0],
        [0]
    ])
    next_state = np.array([
        [[1, 0, 0]],
        [[1, 0, 0]]
    ])

    b = BatchStep(state, action, reward, done, next_state)

    episode = [a, b]

    dataset = SARSGridDataset(episode)

    state, action, reward, done, reset, next_state = dataset[0]
    print(state, action, reward, done, reset)

    state, action, reward, done, reset, next_state = dataset[1]
    print(state, action, reward, done, reset)

    state, action, reward, done, reset, next_state = dataset[2]
    print(state, action, reward, done, reset)

    state, action, reward, done, reset, next_state = dataset[3]
    print(state, action, reward, done, reset)


def test_bandit():
    env = gym.make('SimpleGrid-v2', n=10, map_string="""
    [
    [S, T(1.0)]
    ]
    """)

    obs = env.reset()
    critic = DiscreteQTable((env.height, env.width), 4)
    policy = QPolicy(critic, 4, EpsilonGreedyProperDiscreteDist, epsilon=0.05)

    episode, entropy = batch_episode(env, policy, max_rollout_len=50, render=True)
    dataset = SARSGridDataset(episode)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, pin_memory=True, num_workers=1)

    for state, action, reward, done, reset, next_state in loader:
        pass


def test_gridworld():
    env = gym.make('SimpleGrid-v2', n=10, map_string="""
    [
    [S, E, E, E, E, E, E, E],
    [S, E, E, E, E, E, E, T(1.0)]
    ]
    """)

    obs = env.reset()
    critic = DiscreteQTable((env.height, env.width), 4)
    policy = QPolicy(critic, 4, EpsilonGreedyProperDiscreteDist, epsilon=0.05)

    episode, entropy = batch_episode(env, policy, max_rollout_len=50, render=True)
    dataset = SARSGridDataset(episode)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, pin_memory=True, num_workers=4)

    for state, action, reward, done, reset, next_state in loader:
        pass
