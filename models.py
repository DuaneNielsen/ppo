import torch
from torch import nn
from torch.nn import functional as NN
import copy
import logging
from messages import ModuleHandler


class MultiPolicyNet(nn.Module):
    def __init__(self, features, action_map, hidden=200):
        super().__init__()
        self.features = features
        self.action_map = torch.tensor(action_map)
        self.actions = len(action_map)

        self.l1 = nn.Linear(features, hidden)
        self.l2 = nn.Linear(hidden, self.actions)

    def forward(self, observation):
        hidden = torch.selu(self.l1(observation))
        hidden = self.l2(hidden)
        return NN.log_softmax(hidden, dim=1)

    def sample(self, action_prob):
        index = torch.distributions.Categorical(logits=action_prob).sample()
        action_map = self.action_map.expand(index.size(0), -1)
        gym_action = action_map.take(index)
        return index, gym_action

    def max_action(self, action_logprob):
        probs = torch.exp(action_logprob)
        index = torch.argmax(probs, dim=1)
        gym_action = self.action_map[index]
        return gym_action


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
        return mu, sigma

    def sample(self, mu, sigma):
        return torch.distributions.Normal(mu, sigma).sample()

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
        return mu, sigma

    def sample(self, mu, sigma):
        return torch.distributions.Normal(mu, sigma).sample()

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