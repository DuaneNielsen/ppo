from __future__ import print_function
import os

import util
from models import PPOWrap
from rollout import rollout_policy

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['GPU_DEBUG'] = '0'

import torch
from torch.utils.data import DataLoader
from util import timeit
import math
import configs
from data import RolloutDatasetBase


def ppo_loss(newprob, oldprob, advantage, clip=0.2):
    ratio = newprob / oldprob

    clipped_ratio = ratio.clamp(1.0 - clip, 1.0 + clip)
    clipped_step = clipped_ratio * advantage
    full_step = ratio * advantage
    min_step = torch.stack((full_step, clipped_step), dim=1)
    min_step, clipped = torch.min(min_step, dim=1)

    if config.debug:
        print(f'ADVTG {advantage[0].data}')
        print(f'NEW_P {newprob[0].data}')
        print(f'OLD_P {oldprob[0].data}')
        print(f'RATIO {ratio[0].data}')
        print(f'CLIP_ {clipped_step[0].data}')

    min_step *= -1
    return min_step.mean()


@timeit
def train_policy(policy, rollout_dataset, optim, config):
    policy = policy.train()
    policy = policy.to(config.device)

    batches = math.floor(len(rollout_dataset) / config.max_minibatch_size) + 1
    batch_size = math.floor(len(rollout_dataset) / batches)
    steps_per_batch = math.floor(12 / batches) if math.floor(12 / batches) > 0 else 1
    config.tb.add_scalar('batches', batches, config.tb_step)

    rollout_loader = DataLoader(rollout_dataset, batch_size=batch_size, shuffle=True)
    batches_p = 0
    for i, (observation, action, reward, advantage) in enumerate(rollout_loader):
        batches_p += 1
        for step in range(steps_per_batch):

            observation = observation.to(config.device)
            advantage = advantage.float().to(config.device)
            action = action.squeeze().to(config.device)
            optim.zero_grad()

            if config.debug:
                print(f'ACT__ {action[0].data}')

            new_logprob = policy(observation.squeeze().view(-1, policy.features)).squeeze()
            new_prob = torch.exp(torch.distributions.Categorical(logits=new_logprob).log_prob(action))
            new_logprob.retain_grad()
            old_logprob = policy(observation.squeeze().view(-1, policy.features), old=True).squeeze()
            old_prob = torch.exp(torch.distributions.Categorical(logits=old_logprob).log_prob(action))
            policy.backup()

            loss = ppo_loss(new_prob, old_prob, advantage, clip=0.2)
            loss.backward()
            optim.step()

            if config.debug:
                updated_logprob = policy(observation.squeeze().view(-1, policy.features)).squeeze()
                print(f'CHNGE {( torch.exp(updated_logprob) - torch.exp(new_logprob) ).data[0]}')
                print(f'NEW_G {torch.exp(new_logprob.grad.data[0])}')

            if config.device is 'cuda':
                config.tb.add_scalar('memory_allocated', torch.cuda.memory_allocated(), config.tb_step)
                config.tb.add_scalar('memory_cached', torch.cuda.memory_cached(), config.tb_step)
    print(f'processed {batches_p} batches')
    if gpu_profile:
        gpu_profile(frame=sys._getframe(), event='line', arg=None)


if __name__ == '__main__':

    print('Starting')

    #env_config = configs.AlphaDroneRacer()
    #env_config = configs.Bouncer()
    env_config = configs.LunarLander()
    config = util.Init(env_config.gym_env_string)

    gpu_profile = False
    if gpu_profile:
        from util import gpu_profile
        import sys

        sys.settrace(gpu_profile)

    print(f'Loaded {env_config.gym_env_string}')

    policy_net = PPOWrap(env_config.features, env_config.action_map, env_config.hidden)
    if config.resume:
        policy_net.load_state_dict(torch.load('runs/AlphaRacer2D-v0_941/3.wgt'))

    optim = torch.optim.Adam(lr=1e-4, params=policy_net.new.parameters())

    for epoch in range(config.num_epochs):

        #if env_config.adversarial:
            #rollout = rollout_adversarial_policy(policy_net, env_config)
        #else:
        rollout = rollout_policy(config.num_rollouts, policy_net, env_config)

        dataset = RolloutDatasetBase(env_config, rollout)

        config.tb.add_scalar('collected_frames', len(dataset), config.tb_step)
        train_policy(policy_net, dataset, optim, config)
        torch.cuda.empty_cache()
        # gpu_profile(frame=sys._getframe(), event='line', arg=None)
