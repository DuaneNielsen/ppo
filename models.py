import torch
from torch import nn
from torch.nn import functional as NN


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