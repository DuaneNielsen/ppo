import torch
from torch import nn
from torch.nn import functional as NN
from torch.distributions import *
import copy
from messages import ModuleHandler

import logging
logger = logging.getLogger(__name__)

# class MultiPolicyNet(nn.Module):
#     def __init__(self, features, action_map, hidden=200):
#         super().__init__()
#         self.features = features
#         self.action_map = torch.tensor(action_map)
#         self.actions = len(action_map)
#
#         self.l1 = nn.Linear(features, hidden)
#         self.l2 = nn.Linear(hidden, self.actions)
#
#     def forward(self, observation):
#         hidden = torch.selu(self.l1(observation))
#         hidden = self.l2(hidden)
#         return NN.log_softmax(hidden, dim=1)
#
#     def sample(self, action_prob):
#         index = torch.distributions.Categorical(logits=action_prob).sample()
#         action_map = self.action_map.expand(index.size(0), -1)
#         gym_action = action_map.take(index)
#         return index, gym_action
#
#     def max_action(self, action_logprob):
#         probs = torch.exp(action_logprob)
#         index = torch.argmax(probs, dim=1)
#         gym_action = self.action_map[index]
#         return gym_action


class MultiPolicyNet(nn.Module):
    def __init__(self, features, actions, hidden=200):
        super().__init__()
        self.features = features
        self.actions = actions

        self.l1 = nn.Linear(features, hidden)
        self.l2 = nn.Linear(hidden, actions)

    def forward(self, observation):
        hidden = torch.selu(self.l1(observation))
        hidden = self.l2(hidden)
        logprobs = NN.log_softmax(hidden, dim=1)
        return Categorical(logits=logprobs)

    # def sample(self, action_prob):
    #     index = torch.distributions.Categorical(logits=action_prob).sample()
    #     return index

    def max_action(self, action_logprob):
        probs = torch.exp(action_logprob)
        index = torch.argmax(probs, dim=1)
        return index


class QMLP(nn.Module):
    def __init__(self, features, actions, hidden):
        super().__init__()
        self.features = features
        self.actions = actions

        self.l1 = nn.Linear(features + actions, hidden)
        self.l2 = nn.Linear(hidden, 1)

    def forward(self, state, action):
        state = torch.cat((state, action), dim=1)
        hidden = torch.relu(self.l1(state))
        return self.l2(hidden)


class GreedyDiscreteDist:
    def __init__(self, probs):
        self.probs = probs
        if len(self.probs.shape) == 1:
            self.probs = self.probs.unsqueeze(0)

    def sample(self):
        return torch.argmax(self.probs, dim=1)

    # not sure what this should actually be, below is entropy of a random draw
    def entropy(self):
        return torch.sum(- self.probs * torch.log2(self.probs))

    # this is not correct, the probs are 0 or 1
    def logprob(self, action):
        return torch.log(self.probs[torch.arange(self.probs.size(0)), action])


class OneDistOnly(Exception):
    pass


class EpsilonGreedyDiscreteDist:
    def __init__(self, probs, epsilon=0.05):
        if len(probs.shape) == 1:
            self.probs = probs.unsqueeze(0)
        else:
            self.probs = probs
        if self.probs.size(0) != 1:
            logger.error('Only one discrete probability distribution at a time is currently supported')
            raise OneDistOnly
        self.epsilon = epsilon

        e = self.epsilon / (self.probs.size(1) - 1)
        max = torch.argmax(self.probs, dim=1)
        self.p = torch.ones_like(self.probs) * e
        self.p[torch.arange(self.p.size(0)), max] = 1.0 - self.epsilon

    def sample(self):
        return torch.multinomial(self.p.squeeze(0), 1).unsqueeze(0)

    def entropy(self):
        return torch.sum(- self.probs * torch.log2(self.probs))

    def logprob(self, action):
        return torch.log(self.p[0, action])


class GreedyPolicySlow(nn.Module):
    def __init__(self, qf):
        super().__init__()
        self.qf = qf
        self.actions = torch.eye(self.qf.actions)

    def forward(self, state):
        batch_size = state.size(0)
        states = state.unsqueeze(2).expand(-1, -1, self.qf.actions).unbind(2)
        actions = self.actions.unsqueeze(0).expand(batch_size, -1, -1).unbind(2)

        values = []

        for a in range(self.qf.actions):
            values.append(self.qf(states[a], actions[a]))

        values = torch.stack(values, dim=1).squeeze(2)

        probs = torch.softmax(values, dim=1)
        return GreedyDiscreteDist(probs)


class ValuePolicy(nn.Module):
    def __init__(self, qf, dist_class, **kwargs):
        super().__init__()
        self.qf = qf
        self.actions = torch.eye(self.qf.actions)
        self.dist_class = dist_class
        self.kwargs = kwargs

    def forward(self, state):
        batch_size = state.size(0)
        input_size = state.shape[1:]

        # yeah, this took a bit of work to figure..

        states = state.unsqueeze(1).expand(-1, 4, -1)
        states = states.reshape(batch_size * self.qf.actions, *input_size)

        actions = self.actions.unsqueeze(0).expand(batch_size, -1, -1)
        actions = actions.reshape(batch_size * self.qf.actions, self.qf.actions)

        values = self.qf(states, actions)
        values = values.reshape(batch_size, self.qf.actions)

        probs = torch.softmax(values, dim=1)

        return self.dist_class(probs, **self.kwargs)


class ExplodedGradient(Exception):
    pass


class MultiPolicyNetContinuous(nn.Module):
    def __init__(self, features, actions, hidden=200, scale=1.0, min_sigma=1e-3):
        super().__init__()
        self.features = features
        self.actions = actions
        self.scale = scale
        self.min_sigma = min_sigma

        self.l1 = nn.Linear(features, hidden)
        self.l2 = nn.Linear(hidden, actions * 2)

    def forward(self, observation):
        inter = self.l1(observation)
        hidden = torch.selu(inter)
        if torch.isnan(hidden).any():
            raise ExplodedGradient
        action = self.l2(hidden)
        if torch.isnan(action).any():
            raise ExplodedGradient
        mu, sigma = torch.split(action, self.actions, dim=1)
        mu = torch.tanh(mu)
        sigma = (torch.sigmoid(sigma) * self.scale) + self.min_sigma

        return Normal(mu, sigma)

    # def sample(self, mu, sigma):
    #     return torch.distributions.Normal(mu, sigma).sample()

    def max_action(self, mu, sigma):
        return mu


class MultiPolicyNetContinuousV2(nn.Module):
    def __init__(self, features, actions, hidden=200, scale=1.0, min_sigma=1e-3):
        super().__init__()
        self.features = features
        self.actions = actions
        self.scale = scale
        self.min_sigma = min_sigma

        self.l1_mu = nn.Linear(features, hidden)
        self.l2_mu = nn.Linear(hidden, actions)

        self.l1_sigma = nn.Linear(features, hidden)
        self.l2_sigma = nn.Linear(hidden, actions)

    def forward(self, observation):
        hidden_mu = torch.selu(self.l1_mu(observation))
        mu = torch.tanh(self.l2_mu(hidden_mu))

        hidden_sigma = torch.selu(self.l1_sigma(observation))
        sigma = (torch.sigmoid(self.l2_sigma(hidden_sigma)) * self.scale) + self.min_sigma
        return Normal(mu, sigma)

    # def sample(self, mu, sigma):
    #     return torch.distributions.Normal(mu, sigma).sample()

    def max_action(self, mu, sigma):
        return mu


class PPOWrapModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.old = copy.deepcopy(model)
        self.new = model
        self.features = model.features

    def forward(self, input, old=False):
        if old:
            return self.old(input)
        else:
            return self.new(input)

    def sample(self, *args):
        return self.new.sample(*args)

    def backup(self):
        self.old.load_state_dict(self.new.state_dict())


# this is needed to make the model serializable
# a small price to pay for jsonpickle messages
ModuleHandler.handles(PPOWrapModel)


class PPOWrap(nn.Module):
    def __init__(self, features, action_map, hidden=200):
        super().__init__()
        self.old = MultiPolicyNet(features, action_map, hidden)
        self.new = MultiPolicyNet(features, action_map, hidden)
        self.features = features

    def forward(self, input, old=False):
        if old:
            return self.old(input)
        else:
            return self.new(input)

    def sample(self, action_prob):
        return self.new.sample(action_prob)

    def backup(self):
        self.old.load_state_dict(self.new.state_dict())


# make it transmittable
ModuleHandler.handles(PPOWrap)