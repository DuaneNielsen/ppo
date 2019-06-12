import torch
from torch import nn
from torch.nn import functional as NN
from torch.distributions import *
import copy
from messages import ModuleHandler
from torch.autograd import Variable

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
        self.l2 = nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, 1, bias=False)

    def forward(self, state, action):
        state = torch.cat((state, action), dim=1)
        hidden = torch.relu(self.l1(state))
        hidden = torch.relu(self.l2(hidden))
        return self.output(hidden).squeeze()


class LinearQ(nn.Module):
    def __init__(self, features, actions):
        super().__init__()
        self.features = features
        self.actions = actions

        self.l1 = nn.Linear(features + actions, 1, bias=False)

    def forward(self, state, action):
        state = torch.cat((state, action), dim=1)
        return self.l1(state).squeeze()


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


class GreedyDiscreteDist:
    def __init__(self, probs):
        self.probs = probs
        if len(self.probs.shape) == 1:
            self.probs = self.probs.unsqueeze(0)

    def sample(self):
        """
        Greedy "sampling"
        :return: the maximum action
        """
        one_hot = torch.zeros_like(self.probs)
        one_hot[torch.arange(self.probs.size(0)), torch.argmax(self.probs, dim=1)] = 1.0
        return one_hot

    # not sure what this should actually be, below is entropy of a random draw
    def entropy(self):
        return torch.sum(- self.probs * torch.log2(self.probs))

    # this is not correct, the probs are 0 or 1
    def logprob(self, action):
        return torch.log(torch.sum(self.probs * action, dim=1))


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
        return OneHotCategorical(self.p.squeeze(0)).sample()

    def entropy(self):
        return torch.sum(- self.probs * torch.log2(self.probs))

    def logprob(self, action):
        probs = torch.sum(self.p * action, dim=1)
        return torch.log(probs)


class ValuePolicy(nn.Module):
    def __init__(self, qf, dist_class, **kwargs):
        super().__init__()
        self.qf = qf
        self.actions = torch.eye(self.qf.actions)
        self.dist_class = dist_class
        self.kwargs = kwargs

    def parameters(self, recurse=True):
        return self.qf.parameters(recurse)

    def forward(self, state):
        batch_size = state.size(0)
        input_size = state.shape[1:]

        # yeah, this took a bit of work to figure..
        # compute the value of all actions from a given state

        states = state.unsqueeze(1).expand(-1, self.qf.actions, -1)
        states = states.reshape(batch_size * self.qf.actions, *input_size)

        actions = self.actions.unsqueeze(0).expand(batch_size, -1, -1)
        actions = actions.reshape(batch_size * self.qf.actions, self.qf.actions)

        values = self.qf(states, actions)
        values = values.reshape(batch_size, self.qf.actions)

        #sum = torch.sum(values, dim=1)
        #sum = sum.unsqueeze(1).expand(-1, self.qf.actions)
        #probs = torch.div(values, sum)
        probs = torch.softmax(values, dim=1)

        return self.dist_class(probs, **self.kwargs)


ModuleHandler.handles(ValuePolicy)


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

    def parameters(self, recurse=True):
        return self.new.parameters(recurse)

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


class DefaultTransform:
    def __call__(self, observation, dtype=torch.float32, insert_batch=False):
        """
        env -> model tensor
        :param observation: the raw observation
        :param insert_batch: add a batch dimension to the front
        :return: tensor in shape (batch, dims)
        """
        if insert_batch:
            return torch.from_numpy(observation).to(dtype=dtype).unsqueeze(0)
        else:
            return torch.from_numpy(observation).to(dtype=dtype)


class InfinityException(Exception):
    pass


class ContinousActionTransform:
    def __call__(self, action):
        action = action.squeeze()
        if torch.isnan(action).any().item() == 1:
            raise InfinityException
        return action.numpy()

    def invert(self, action, dtype):
        return torch.from_numpy(action).to(dtype=dtype)


class DiscreteActionTransform:
    def __init__(self, action_map):
        """
        Takes a single action from model and converts it to an integer mapped
        to the environments action space
        :param action_map:
        """
        self.action_map = torch.tensor(action_map)

        self.reverse = [0] * (max(action_map) + 1)

        for i, item in enumerate(action_map):
            self.reverse[item] = i

    def __call__(self, index):
        """
        model -> environment
        :param index:
        :return:
        """
        action_map = self.action_map.expand(index.size(0), -1)
        action = action_map.take(index)
        return action.squeeze().item()

    def invert(self, action, dtype=None):
        """
        env -> model
        converts the action on the environment action space back to the model action space
        :param action:
        :return:
        """
        return torch.tensor([self.reverse[action]])


class OneHotDiscreteActionTransform:
    def __init__(self, action_map):
        self.action_map = torch.tensor(action_map)

        self.reverse = [0] * (max(action_map) + 1)

        for i, item in enumerate(action_map):
            self.reverse[item] = i

    def __call__(self, action):
        """
        model -> environment
        :return:
        """
        return self.action_map[torch.argmax(action.squeeze())].item()

    def invert(self, action, dtype):
        """
        converts one action from env -> model
        :param action:
        :param dtype:
        :return:
        """
        one_hot = torch.zeros(self.action_map.size(0), dtype=dtype)
        index = self.reverse[action]
        one_hot[index] = 1.0
        return one_hot