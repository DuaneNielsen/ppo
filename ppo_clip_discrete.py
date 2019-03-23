from __future__ import print_function
import os

import util
from configs import AlphaDroneRacer
from models import PPOWrap

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['GPU_DEBUG'] = '0'

import torch
from torch.utils.data import DataLoader
import gym
from util import UniImageViewer, timeit
import math
import configs
import gym_duane

@timeit
def rollout_adversarial_policy(policy, env_config):
    policy = policy.eval()
    policy = policy.to('cpu')
    rollout_dataset = env_config.construct_dataset()
    reward_total = 0
    env = gym.make(env_config.gym_env_string)
    v1 = UniImageViewer(env_config.gym_env_string + ' player 1', (200, 160))
    v2 = UniImageViewer(env_config.gym_env_string + ' player 2', (200, 160))

    for i in range(num_rollouts):

        episode_length = 0

        observation_t0 = env.reset()
        action = env_config.default_action
        observation_t1, rewards, done, info = env.step((action, action))
        observations = [env_config.prepro(t1, t0) for t1, t0 in zip(observation_t1, observation_t0)]
        observation_t0 = observation_t1
        done = False

        while not done:
            # take an action on current observation and record result
            observation_tensors = [rollout_dataset.transform(o) for o in observations]
            obs_stack = torch.stack(observation_tensors, dim=0)
            action_prob = policy(obs_stack)
            index, action = policy.sample(action_prob)
            actions = [a.item() for a in action.chunk(action.size(0), dim=0)]
            indices = [i.item() for i in index.chunk(index.size(0), dim=0)]


            observation_t1, rewards, done, info = env.step(actions)

            done = done or episode_length > env_config.max_rollout_len
            reward_total += sum(rewards)
            episode_length += 1

            for i, (o, a, r) in enumerate(zip(observations, indices, rewards)):
                rollout_dataset.append(o, a, r, done, episode=i)

            # compute the observation that resulted from our action
            observations = [env_config.prepro(t1, t0) for t1, t0 in zip(observation_t1, observation_t0)]
            observation_t0 = observation_t1

            if view_games:
                v1.render(observations[0])
                v2.render(observations[1])
                env.render(mode='human')

        # more monitoring
        config.tb.add_scalar('reward', reward_total, config.tb_step)
        reward_total = 0
        config.tb.add_scalar('epi_len', episode_length, config.tb_step)
        config.tb_step += 1

    # save the file every so often
    if epoch % 20 == 0:
        torch.save(policy.state_dict(), config.rundir + '/vanilla.wgt')

    rollout_dataset.end_rollout()
    return rollout_dataset


@timeit
def rollout_policy(policy, env_config):

    policy = policy.eval()
    policy = policy.to('cpu')
    rollout_dataset = env_config.construct_dataset()
    reward_total = 0
    v = UniImageViewer(env_config.gym_env_string, (200, 160))
    env = gym.make(env_config.gym_env_string)


    for i in range(num_rollouts):

        episode_length = 0

        observation_t0 = env.reset()
        action = env_config.default_action
        observation_t1, reward, done, info = env.step(action)
        observation = env_config.prepro(observation_t1, observation_t0)
        observation_t0 = observation_t1
        done = False

        while not done:
            # take an action on current observation and record result
            observation_tensor = rollout_dataset.transform(observation, insert_batch=True)
            action_prob = policy(observation_tensor)
            index, action = policy.sample(action_prob)

            observation_t1, reward, done, info = env.step(action.squeeze().item())

            done = done or episode_length > env_config.max_rollout_len
            reward_total += reward
            episode_length += 1

            rollout_dataset.append(observation, reward, index, done)

            # compute the observation that resulted from our action
            observation = env_config.prepro(observation_t1, observation_t0)
            observation_t0 = observation_t1

            if view_games:
                env.render(mode='human')
            if view_obs:
                v.render(observation)

        # more monitoring
        config.tb.add_scalar('reward', reward_total, config.tb_step)
        global best_reward
        if not best_reward or reward_total > best_reward:
            best_reward = reward_total
            save_epoch = config.tb_step // save_freq
            torch.save(policy.state_dict(), config.rundir + f'/{save_epoch}.wgt')

        if config.tb_step % save_freq == 0:
            best_reward = None
        reward_total = 0

        config.tb.add_scalar('epi_len', episode_length, config.tb_step)
        config.tb_step += 1

    torch.save(policy.state_dict(), config.rundir + f'/latest.wgt')

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
    steps_per_batch = math.floor(12 / batches) if math.floor(12 / batches) > 0 else 1
    config.tb.add_scalar('batches', batches, config.tb_step)

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

            if device is 'cuda':
                config.tb.add_scalar('memory_allocated', torch.cuda.memory_allocated(), config.tb_step)
                config.tb.add_scalar('memory_cached', torch.cuda.memory_cached(), config.tb_step)
    print(f'processed {batches_p} batches')
    if gpu_profile:
        gpu_profile(frame=sys._getframe(), event='line', arg=None)


if __name__ == '__main__':

    print('Starting')

    gpu_profile = False
    if gpu_profile:
        from util import gpu_profile
        import sys

        sys.settrace(gpu_profile)



    num_epochs = 6000
    num_rollouts = 60
    collected_rollouts = 0
    device = 'cpu' if torch.cuda.is_available() else 'cpu'
    max_minibatch_size = 400000
    resume = False
    view_games = False
    view_obs = False
    debug = False
    save_freq = 1000
    best_reward = None

    #env_config = configs.AlphaDroneRacer()
    #env_config = configs.Bouncer()
    env_config = configs.LunarLander()
    config = util.Init(env_config.gym_env_string)

    print(f'Loaded {env_config.gym_env_string}')

    policy_net = PPOWrap(env_config.features, env_config.action_map, env_config.hidden)
    if resume:
        policy_net.load_state_dict(torch.load('runs/AlphaRacer2D-v0_941/3.wgt'))

    optim = torch.optim.Adam(lr=1e-4, params=policy_net.new.parameters())

    for epoch in range(num_epochs):

        if env_config.adversarial:
            rollout_dataset = rollout_adversarial_policy(policy_net, env_config)
        else:
            rollout_dataset = rollout_policy(policy_net, env_config)

        config.tb.add_scalar('collected_frames', len(rollout_dataset), config.tb_step)
        train_policy(policy_net, rollout_dataset, optim, device)
        torch.cuda.empty_cache()
        # gpu_profile(frame=sys._getframe(), event='line', arg=None)
