from __future__ import print_function
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['GPU_DEBUG'] = '0'

import torch
from torch.utils.data import DataLoader
from util import timeit
import math
import logging
import duallog
gpu_profile = False
if gpu_profile:
    from util import gpu_profile
    import sys
    sys.settrace(gpu_profile)


def ppo_loss(newprob, oldprob, advantage, clip=0.2):
    ratio = newprob / oldprob

    clipped_ratio = ratio.clamp(1.0 - clip, 1.0 + clip)
    clipped_step = clipped_ratio * advantage
    full_step = ratio * advantage
    min_step = torch.stack((full_step, clipped_step), dim=1)
    min_step, clipped = torch.min(min_step, dim=1)

    logging.debug(f'ADVTG {advantage[0].data}')
    logging.debug(f'NEW_P {newprob[0].data}')
    logging.debug(f'OLD_P {oldprob[0].data}')
    logging.debug(f'RATIO {ratio[0].data}')
    logging.debug(f'CLIP_ {clipped_step[0].data}')

    min_step *= -1
    return min_step.mean()


def ppo_loss_log(newlogprob, oldlogprob, advantage, clip=0.2):
    ratio = torch.exp(newlogprob - oldlogprob)

    clipped_ratio = ratio.clamp(1.0 - clip, 1.0 + clip)
    clipped_step = clipped_ratio * advantage.unsqueeze(1)
    full_step = ratio * advantage.unsqueeze(1)
    min_step = torch.stack((full_step, clipped_step), dim=1)
    min_step, clipped = torch.min(min_step, dim=1)

    logging.debug(f'ADVTG {advantage[0].data}')
    #logging.debug(f'NEW_P {newprob[0].data}')
    #logging.debug(f'OLD_P {oldprob[0].data}')
    logging.debug(f'RATIO {ratio[0].data}')
    logging.debug(f'CLIP_ {clipped_step[0].data}')

    min_step *= -1
    return min_step.mean()


@timeit
def train_policy(policy, rollout_dataset, config, device='cpu'):

    optim = torch.optim.Adam(lr=1e-4, params=policy.new.parameters())
    policy = policy.train()
    policy = policy.to(device)

    batches = math.floor(len(rollout_dataset) / config.max_minibatch_size) + 1
    batch_size = math.floor(len(rollout_dataset) / batches)
    steps_per_batch = math.floor(12 / batches) if math.floor(12 / batches) > 0 else 1

    rollout_loader = DataLoader(rollout_dataset, batch_size=batch_size, shuffle=True)
    batches_p = 0
    for i, (observation, action, reward, advantage) in enumerate(rollout_loader):
        batches_p += 1
        for step in range(steps_per_batch):

            observation = observation.to(device)
            advantage = advantage.float().to(device)
            action = action.squeeze().to(device)
            optim.zero_grad()

            logging.debug(f'ACT__ {action[0].data}')

            new_logprob = policy(observation.squeeze().view(-1, policy.features)).squeeze()
            new_prob = torch.exp(torch.distributions.Categorical(logits=new_logprob).log_prob(action))
            new_logprob.retain_grad()
            old_logprob = policy(observation.squeeze().view(-1, policy.features), old=True).squeeze()
            old_prob = torch.exp(torch.distributions.Categorical(logits=old_logprob).log_prob(action))
            policy.backup()

            loss = ppo_loss(new_prob, old_prob, advantage, clip=0.2)
            loss.backward()
            optim.step()

            updated_logprob = policy(observation.squeeze().view(-1, policy.features)).squeeze()
            logging.debug(f'CHNGE {(torch.exp(updated_logprob) - torch.exp(new_logprob)).data[0]}')
            logging.debug(f'NEW_G {torch.exp(new_logprob.grad.data[0])}')

    logging.info(f'processed {batches_p} batches')
    if config.gpu_profile:
        gpu_profile(frame=sys._getframe(), event='line', arg=None)


# need to get rid of this somehow
def format_observation(observation, features):
    return observation.squeeze().view(-1, features)


def train_ppo_continuous(policy, dataset, config, device='cpu'):
    optim = torch.optim.Adam(lr=1e-4, params=policy.new.parameters())
    policy = policy.train()
    policy = policy.to(device)

    batches = math.floor(len(dataset) / config.max_minibatch_size) + 1
    batch_size = math.floor(len(dataset) / batches)
    steps_per_batch = math.floor(12 / batches) if math.floor(12 / batches) > 0 else 1

    rollout_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    batches_p = 0
    for i, (observation, action, reward, advantage) in enumerate(rollout_loader):
        batches_p += 1
        for step in range(steps_per_batch):

            observation = observation.to(device)
            advantage = advantage.float().to(device)
            action = action.squeeze().to(device)
            optim.zero_grad()

            # if config.debug:
            #     print(f'ACT__ {action[0].data}')

            mu, sigma = policy(format_observation(observation, policy.features))
            new_prob = torch.distributions.Normal(mu, sigma).log_prob(action).clamp(max=0.0)
            new_prob.retain_grad()
            old_mu, old_sigma = policy(format_observation(observation, policy.features), old=True)
            old_prob = torch.distributions.Normal(old_mu, old_sigma).log_prob(action).clamp(max=0.0)
            policy.backup()

            loss = ppo_loss_log(new_prob, old_prob, advantage, clip=0.2)
            loss.backward()
            optim.step()

            # if config.debug:
            #     updated_logprob = policy(observation.squeeze().view(-1, policy.features)).squeeze()
            #     print(f'CHNGE {(torch.exp(updated_logprob) - torch.exp(new_logprob)).data[0]}')
            #     print(f'NEW_G {torch.exp(new_logprob.grad.data[0])}')

    logging.info(f'processed {batches_p} batches')
    if config.gpu_profile:
        gpu_profile(frame=sys._getframe(), event='line', arg=None)
