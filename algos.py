from __future__ import print_function
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['GPU_DEBUG'] = '0'

import torch
from torch import tensor

from configs import AlgoConfig
from util import Converged
from torch.utils.data import DataLoader
import logging
gpu_profile = False
if gpu_profile:
    from util import gpu_profile
    import sys
    sys.settrace(gpu_profile)

logger = logging.getLogger(__name__)
from time import time
from data import *
from models import *


def ppo_loss(newprob, oldprob, advantage, clip=0.2):
    ratio = newprob / oldprob

    clipped_ratio = ratio.clamp(1.0 - clip, 1.0 + clip)
    clipped_step = clipped_ratio * advantage
    full_step = ratio * advantage
    min_step = torch.stack((full_step, clipped_step), dim=1)
    min_step, clipped = torch.min(min_step, dim=1)

    logger.debug(f'ADVTG {advantage[0].data}')
    logger.debug(f'NEW_P {newprob[0].data}')
    logger.debug(f'OLD_P {oldprob[0].data}')
    logger.debug(f'RATIO {ratio[0].data}')
    logger.debug(f'CLIP_ {clipped_step[0].data}')

    min_step *= -1
    return min_step.mean()


def ppo_loss_log(newlogprob, oldlogprob, advantage, clip=0.2):
    log_ratio = (newlogprob - oldlogprob)
    # clamp the log to stop infinities (85 is for 32 bit floats)
    log_ratio.clamp_(min=-10.0, max=10.0)
    ratio = torch.exp(log_ratio)

    clipped_ratio = ratio.clamp(1.0 - clip, 1.0 + clip)
    clipped_step = clipped_ratio * advantage.unsqueeze(1)
    full_step = ratio * advantage.unsqueeze(1)
    min_step = torch.stack((full_step, clipped_step), dim=1)
    min_step, clipped = torch.min(min_step, dim=1)

    # logger.info(f'mean advantage : {advantage.mean()}')
    # logger.info(f'mean newlog    : {newlogprob.mean()}')
    # logger.info(f'mean oldlob    : {oldlogprob.mean()}')
    # logger.info(f'mean log_ratio : {log_ratio.mean()}')
    # logger.info(f'mean ratio     : {ratio.mean()}')
    # logger.info(f'mean clip ratio: {clipped_ratio.mean()}')
    # logger.info(f'mean clip step : {clipped_step.mean()}')

    min_step *= -1
    return min_step.mean()


class PurePPOClipConfig(AlgoConfig):
    def __init__(self, optimizer, ppo_steps_per_batch):
        super().__init__(PurePPOClip)
        self.optimizer = optimizer
        self.ppo_steps_per_batch = ppo_steps_per_batch


class PurePPOClip:
    def __call__(self, policy, exp_buffer, config, device='cpu'):

        optim = config.algo.optimizer.construct(policy.parameters())
        policy = policy.train()
        policy = policy.to(device)
        dataset = SARAdvantageDataset(exp_buffer, discount_factor=config.discount_factor, state_transform=config.transform,
                                      action_transform=config.action_transform, precision=config.precision)
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

        batches_p = 0
        for observation, action, reward, advantage in loader:
            batches_p += 1
            for step in range(config.algo.ppo_steps_per_batch):

                #todo catergorical distrubution loss.backward() super slow (pytorch problem)
                optim.zero_grad()

                new_dist = policy(observation)
                new_logprob = new_dist.log_prob(action)
                new_logprob.retain_grad()

                old_dist = policy(observation, old=True)
                old_logprob = old_dist.log_prob(action)
                policy.backup()

                loss = ppo_loss_log(new_logprob, old_logprob, advantage, clip=0.2)
                #logging.info(f'loss {loss.item()}')

                starttime = time()
                loss.backward()
                endtime = time()

                optim.step()

                logger.debug(endtime - starttime)

        logging.info(f'processed {batches_p} batches')
        if config.gpu_profile:
            gpu_profile(frame=sys._getframe(), event='line', arg=None)

        return policy


class NoOptimizer(Exception):
    pass


class OptimizerConfig:
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs

    def construct(self, parameters):
        if self.name == 'Adam':
            optim = torch.optim.Adam(params=parameters, **self.kwargs)

        elif self.name == 'SGD':
            optim = torch.optim.SGD(params=parameters, **self.kwargs)
        else:
            raise NoOptimizer

        return optim


class OneStepTDConfig(AlgoConfig):
    def __init__(self, optimizer, min_change=2e-4, detections=5, detection_window=8):
        super().__init__(OneStepTD)
        self.optimizer = optimizer
        self.min_change = min_change
        self.detections = detections
        self.detection_window = detection_window


class OneStepTD:
    def __call__(self, critic, exp_buffer, config, device='cpu'):

        one_step_config = config.algo
        greedy_policy = ValuePolicy(critic, GreedyDiscreteDist)
        optim = one_step_config.optimizer.construct(critic.parameters())
        dataset = SARSDataset(exp_buffer, state_transform=config.transform, action_transform=config.action_transform)
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
        c = Converged(one_step_config.min_change, detections=one_step_config.detections,
                      detection_window=one_step_config.detection_window)

        loss = torch.tensor([-1.0])

        while not c.converged(loss.item()):
            logger.info(f'loss: {loss.item()}')

            for state, action, reward, next_state, done in loader:

                zero_if_terminal = (~done).to(next_state.dtype)
                next_action = greedy_policy(next_state).sample()
                next_value = critic(next_state, next_action)
                target = reward + zero_if_terminal * config.discount_factor * next_value

                optim.zero_grad()
                predicted = critic(state, action)
                loss = torch.mean((target - predicted) ** 2)
                loss.backward()
                optim.step()

        # return an epsilon greedy policy as actor
        return ValuePolicy(critic, EpsilonGreedyDiscreteDist, epsilon=0.2)


# todo need to get rid of this somehow, just format the observation correctly, or use the transform from the config
def format_observation(observation, features):
    return observation.squeeze().view(-1, features)


class InprobableActionException(Exception):
    pass


def train_ppo_continuous(policy, dataset, config, device='cpu'):

    optim = optimizer(config, policy)
    policy = policy.train()
    policy = policy.to(device)

    # batches = math.floor(len(dataset) / config.max_minibatch_size) + 1
    # batch_size = math.floor(len(dataset) / batches)
    # steps_per_batch = math.floor(12 / batches) if math.floor(12 / batches) > 0 else 1

    rollout_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    batches_p = 0
    for i, (observation, action, reward, advantage) in enumerate(rollout_loader):
        batches_p += 1
        for step in range(config.ppo_steps_per_batch):

            observation = observation.to(device)
            advantage = advantage.float().to(device)
            action = action.squeeze().to(device)
            optim.zero_grad()

            # if config.debug:
            #     print(f'ACT__ {action[0].data}')

            mu, sigma = policy(format_observation(observation, policy.features))
            new_dist = torch.distributions.Normal(mu, sigma)
            new_prob = new_dist.log_prob(action)


            #new_prob = torch.distributions.Normal(mu, sigma).log_prob(action).clamp(max=0.0)

            new_prob.retain_grad()
            # if torch.isnan(new_prob).any():
            #     logger.error(new_prob)
            #     logger.error(mu)
            #     logger.error(sigma)
            #     raise InprobableActionException
            old_mu, old_sigma = policy(format_observation(observation, policy.features), old=True)
            old_prob = torch.distributions.Normal(old_mu, old_sigma).log_prob(action)
            #old_prob = torch.distributions.Normal(old_mu, old_sigma).log_prob(action).clamp(max=0.0)
            policy.backup()
            entropy = new_dist.entropy().mean()
            logger.info(f'ENTROPY {entropy.item()}')
            ppo_loss = ppo_loss_log(new_prob, old_prob, advantage, clip=0.2)
            loss = ppo_loss
            loss.backward()
            logger.info(f'LOSS {loss.item()}')
            optim.step()

            # if config.debug:
            #     updated_logprob = policy(observation.squeeze().view(-1, policy.features)).squeeze()
            #     print(f'CHNGE {(torch.exp(updated_logprob) - torch.exp(new_logprob)).data[0]}')
            #     print(f'NEW_G {torch.exp(new_logprob.grad.data[0])}')

    #logging.info(f'processed {batches_p} batches')
    if config.gpu_profile:
        gpu_profile(frame=sys._getframe(), event='line', arg=None)
