from __future__ import print_function
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['GPU_DEBUG'] = '0'

import torch
from torch import tensor
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
import models

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


class NoOptimizer(Exception):
    pass

def optimizer(config, policy):
    if config.optimizer == 'Adam':
        optim = torch.optim.Adam(lr=config.lr, params=policy.new.parameters())

    elif config.optimizer == 'SGD':
        optim = torch.optim.SGD(lr=config.lr, params=policy.new.parameters(), weight_decay=1e-4)
    else:
        raise NoOptimizer

    return optim


class PurePPOClip:
    def __call__(self, policy, exp_buffer, config, device='cpu'):

        optim = optimizer(config, policy)
        policy = policy.train()
        policy = policy.to(device)
        dataset = SARAdvantageDataset(exp_buffer, discount_factor=config.discount_factor, state_transform=config.transform,
                                      action_transform=config.action_transform, precision=config.precision)
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

        batches_p = 0
        for observation, action, reward, advantage in loader:
            batches_p += 1
            for step in range(config.ppo_steps_per_batch):

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


class OneStepTD:
    def __init__(self, config):
        self.q_f = models.QMLP(config.features, len(config.action_map), config.features + len(config.action_map))
        self.greedy_policy = models.GreedyPolicy(self.q_f)

    def __call__(self, policy, exp_buffer, config, device='cpu'):

        optim = optimizer(config, self.q_f.parameters())
        dataset = SARSDataset(exp_buffer)
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

        for _ in range(100):

            for state, action, reward, next_state in loader:
                next_action = self.greedy_policy(next_state).sample()
                target = reward + config.discount * self.q_f(next_state, next_action)

                optim.zero_grad()
                predicted = self.q_f(state, action)
                loss = torch.mean((target - predicted)) ** 2
                loss.backward()
                optim.step()

        return models.EpsilonGreedyPolicy(self.q_f, 0.05)


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
