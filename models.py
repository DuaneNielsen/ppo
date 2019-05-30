import torch
from torch import nn
from torch.nn import functional as NN
import copy


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


class MultiPolicyNetContinuous(nn.Module):
    def __init__(self, features, actions, hidden=200, scale=1.0):
        super().__init__()
        self.features = features
        self.actions = actions
        self.scale = scale

        self.l1 = nn.Linear(features, hidden)
        self.l2 = nn.Linear(hidden, actions * 2)

    def forward(self, observation):
        hidden = torch.selu(self.l1(observation))
        action = self.l2(hidden)
        mu, sigma = torch.split(action, self.actions, dim=1)
        mu = torch.tanh(mu)
        sigma = torch.sigmoid(sigma) * self.scale
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