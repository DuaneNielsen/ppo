from __future__ import print_function
import os

import util
from data import RolloutDataSet
from models import PPOWrap

os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['GPU_DEBUG']='0'

import torch
from torch.utils.data import DataLoader
import gym
from util import UniImageViewer, timeit, tb_step, rundir, tb
import cv2
import statistics
from torchvision.transforms.functional import to_tensor
import numpy as np
import math


@timeit
def rollout_policy(policy, env, downsample, transform):
    policy = policy.eval()
    policy = policy.to('cpu')
    rollout_dataset = RolloutDataSet(discount_factor=0.99, transform=transform)
    reward_total = 0

    for i in range(num_rollouts):

        game_length = 0
        gl = []
        probs = []

        observation_t0 = env.reset()
        observation_t0 = downsample(observation_t0)
        action = default_action
        observation_t1, reward, done, info = env.step(action)
        observation_t1 = downsample(observation_t1)
        observation = observation_t1 - observation_t0
        observation_t0 = observation_t1
        done = False

        while not done:
            # take an action on current observation and record result
            observation_tensor = transform(observation)
            action_prob = policy(observation_tensor)
            index, action = policy.sample(action_prob)
            observation_t1, reward, done, info = env.step(action.squeeze().item())
            reward_total += reward

            rollout_dataset.append(observation, reward, index, done)

            # compute the observation that resulted from our action
            observation_t1 = downsample(observation_t1)
            observation = observation_t1 - observation_t0
            observation_t0 = observation_t1

            # monitoring
            if reward == 0:
                game_length += 1
                if debug:
                    probs.append(torch.exp(action_prob.squeeze()))
            else:
                gl.append(game_length)
                game_length = 0

                if debug:
                    probs = torch.stack(probs)
                    mean = probs.mean(dim=0)
                    print(mean[0].item(), mean[1].item())
                    del probs
                    probs = []

            if view_games:
                v.render(observation)
                env.render(mode='human')

        # more monitoring
        # hooks.execute_test_end(collected_rollouts, num_rollouts, reward_total)
        tb.add_scalar('reward', reward_total, util.tb_step)
        tb.add_scalar('ave_game_len', statistics.mean(gl), util.tb_step)
        reward_total = 0
        util.tb_step += 1

    # save the file every so often
    if epoch % 20 == 0:
        torch.save(policy.state_dict(), util.rundir + '/vanilla.wgt')

    rollout_dataset.normalize()
    return rollout_dataset


def ppo_loss(newprob, oldprob, advantage, clip=0.2):

    ratio = newprob / oldprob

    clipped_ratio = ratio.clamp(1.0 - clip, 1.0 + clip)
    clipped_step = clipped_ratio * advantage
    full_step = ratio * advantage
    min_step = torch.stack((full_step, clipped_step), dim=1)
    min_step, clipped = torch.min(min_step, dim=1)

    if debug:
        print(f'ADVTG {advantage[0].data}')
        print(f'NEW_P {newprob[0].data}')
        print(f'OLD_P {oldprob[0].data}')
        print(f'RATIO {ratio[0].data}')
        print(f'CLIP_ {clipped_step[0].data}')

    min_step *= -1
    return min_step.mean()


@timeit
def train_policy(policy, rollout_dataset, optim, device='cpu'):
    policy = policy.train()
    policy = policy.to(device)

    batches = math.floor(len(rollout_dataset) / max_minibatch_size) + 1
    batch_size = math.floor(len(rollout_dataset) / batches)
    steps_per_batch = math.floor(12 / batches) if math.floor(12/batches) > 0 else 1
    tb.add_scalar('batches', batches, tb_step)

    rollout_loader = DataLoader(rollout_dataset, batch_size=batch_size, shuffle=True)
    batches_p = 0
    for i, (observation, reward, action, advantage) in enumerate(rollout_loader):
        batches_p += 1
        for step in range(steps_per_batch):

            observation = observation.to(device)
            advantage = advantage.float().to(device)
            action = action.squeeze().to(device)
            optim.zero_grad()

            if debug:
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

            if debug:
                updated_logprob = policy(observation.squeeze().view(-1, policy.features)).squeeze()
                print(f'CHNGE {( torch.exp(updated_logprob) - torch.exp(new_logprob) ).data[0]}')
                print(f'NEW_G {torch.exp(new_logprob.grad.data[0])}')

            tb.add_scalar('memory_allocated', torch.cuda.memory_allocated(), tb_step)
            tb.add_scalar('memory_cached', torch.cuda.memory_cached(), tb_step)
    print(f'processed {batches_p} batches')
    #gpu_profile(frame=sys._getframe(), event='line', arg=None)



def downsample_pong(observation):
    greyscale = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    greyscale = cv2.resize(greyscale, downsample_image_size, cv2.INTER_LINEAR)
    return greyscale


class TransformAtari:
    def __init__(self, features):
        self.features = features

    def __call__(self, observation):
        to_tensor(np.expand_dims(observation, axis=2))\
                    .squeeze().unsqueeze(0).view(-1, self.features)

def lunar_lander_transform(observation):
    """
    :param observation: the raw observation
    :return: tensor in shape (batch, dims)
    """
    return torch.from_numpy(observation).float().unsqueeze(0)


def downsample_lunar_lander(observation):
    return observation


class LunarLander:
    def __init__(self):
        self.gym_env_string = 'LunarLander-v2'
        self.features = 8
        self.hidden = 8
        self.action_map = [0, 1, 2, 3]
        self.default_action = 0
        self.prepro = downsample_lunar_lander
        self.transform = lunar_lander_transform


if __name__ == '__main__':

    gpu_profile = False
    if gpu_profile:
        from util import gpu_profile
        import sys
        sys.settrace(gpu_profile)

    max_rollout_len = 3000
    downsample_image_size = (100, 80)
    features = downsample_image_size[0] * downsample_image_size[1]
    default_action = 0
    num_epochs = 6000
    num_rollouts = 10
    collected_rollouts = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_minibatch_size = 40000
    resume = False
    view_games = False
    debug = False

    env_config = LunarLander()

    env = gym.make(env_config.gym_env_string)
    v = UniImageViewer(env_config.gym_env_string, (200, 160))
    # GUIProgressMeter('training_pong')
    policy_net = PPOWrap(env_config.features, env_config.action_map, env_config.hidden)
    if resume:
        policy_net.load_state_dict(torch.load('runs/ppo_multilabel_259/vanilla.wgt'))

    optim = torch.optim.Adam(lr=1e-4, params=policy_net.new.parameters())

    for epoch in range(num_epochs):
        rollout_dataset = rollout_policy(policy_net, env, env_config.prepro, env_config.transform)
        tb.add_scalar('collected_frames', len(rollout_dataset), tb_step)
        train_policy(policy_net, rollout_dataset, optim, device)
        torch.cuda.empty_cache()
        #gpu_profile(frame=sys._getframe(), event='line', arg=None)

    # hooks.execute_epoch_end(epoch, num_epochs)
