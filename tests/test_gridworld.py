import gym
from data.transforms import *
import gym_duane
import configs
from data.data import Step
import numpy as np
from statistics import mean
from torch import nn
from torch.distributions import Categorical


def batch_episode(env, policy, max_rollout_len=4000, v=None, render=False, display_observation=False):

    episode = []
    episode_length = 0
    observation_t0 = env.reset()
    entropy = []
    state = observation_t0

    all_done = False
    while not all_done:
        # take an action on current observation and record result
        action_dist = policy(state)

        entropy.append(action_dist.entropy().mean().item())

        action = action_dist.sample()

        observation_t1, reward, done, info = env.step(action)

        all_done = torch.all(done).item() == 1\
                   or episode_length > max_rollout_len

        if episode is not None:
            state = state.cpu().numpy()
            action = action.cpu().numpy()
            reward = reward.cpu().numpy()
            done = done.cpu().numpy()
            episode.append(Step(state, action, reward, done))
        episode_length += 1

        # compute the observation that resulted from our action
        state = observation_t1

        if render:
            env.render(mode='human')
        if display_observation:
            v.render(state)

    final_entropy = 0
    if episode is not None:
        episode.append(Step(state, np.zeros(state.shape[0]), np.zeros(state.shape[0]), np.ones(state.shape[0])))
        final_entropy = mean(entropy)
    return episode, final_entropy


class QTable(nn.Module):
    def __init__(self, features, actions):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(features, actions))
        self.states = features
        self.actions = actions

    def forward(self, state, action):
        state = state.unsqueeze(2)
        action = action.unsqueeze(1)
        context = torch.matmul(state, action)
        context = context * self.weights
        return torch.sum(context, dim=[1, 2])


class RandomPolicy:
    def __call__(self, state):
        p = torch.ones(state.size(0), 4) / 4
        return Categorical(p)

def test_gridworld():
    env = gym.make('SimpleGrid-v2', n=3, map_string="""
    [
    [S, E, E, T]
    ]
    """)

    obs = env.reset()
    policy = RandomPolicy()

    episode, entropy = batch_episode(env, policy)

    print(len(episode))
    # for _ in range(100):
    #     action = torch.LongTensor(3).random_(0, 3)
    #     obs, reward, done, info = env.step(action)