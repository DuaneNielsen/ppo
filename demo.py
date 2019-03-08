import torch
import argparse
import gym

import models
import time
import pathlib
import inspect
import configs
import gym_duane


def demo(env_config, policy, speed, episodes=3):
    env = gym.make(env_config.gym_env_string)

    for i in range(episodes):

        episode_length = 0

        observation_t0 = env.reset()
        action = env_config.default_action
        observation_t1, reward, done, info = env.step(action)
        observation = env_config.prepro(observation_t1, observation_t0)
        observation_t0 = observation_t1
        done = False

        while not done:
            observation_tensor = env_config.transform(observation, insert_batch=True)
            action_logprob = policy(observation_tensor)
            action = policy.new.max_action(action_logprob)
            observation_t1, reward, done, info = env.step(action.squeeze().item())

            env.render(mode='human')
            done = done or episode_length > env_config.max_rollout_len

            # compute the observation that resulted from our action
            observation = env_config.prepro(observation_t1, observation_t0)
            observation_t0 = observation_t1

            time.sleep(speed)
            episode_length += 1


def demo_adversarial(env_config, policy, speed, episodes=3):
    policy = policy.eval()
    policy = policy.to('cpu')
    reward_total = 0
    env = gym.make(env_config.gym_env_string)

    for i in range(episodes):

        episode_length = 0

        observation_t0 = env.reset()
        action = env_config.default_action
        observation_t1, rewards, done, info = env.step((action, action))
        observations = [env_config.prepro(t1, t0) for t1, t0 in zip(observation_t1, observation_t0)]
        observation_t0 = observation_t1
        done = False

        while not done:
            # take an action on current observation and record result
            observation_tensors = [env_config.transform(o) for o in observations]
            obs_stack = torch.stack(observation_tensors, dim=0)
            action_prob = policy(obs_stack)
            index, action = policy.sample(action_prob)
            actions = [a.item() for a in action.chunk(action.size(0), dim=0)]
            indices = [i.item() for i in index.chunk(index.size(0), dim=0)]

            observation_t1, rewards, done, info = env.step(actions)
            env.render('human')

            done = done or episode_length > env_config.max_rollout_len
            reward_total += sum(rewards)
            episode_length += 1

            # compute the observation that resulted from our action
            observations = [env_config.prepro(t1, t0) for t1, t0 in zip(observation_t1, observation_t0)]
            observation_t0 = observation_t1


if __name__ == '__main__':

    parser = argparse.ArgumentParser('learn to play pong')
    parser.add_argument('--env', default='LunarLander')
    parser.add_argument('--reload', default='config')
    parser.add_argument('--latest', dest='latest', action='store_true')
    parser.add_argument('--speed', type=float, default=0.02)
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--list', dest='list', action='store_true')
    parser.set_defaults(multi=False, ppo=True)
    args = parser.parse_args()

    if args.list:
        for name, obj in inspect.getmembers(configs):
            if inspect.isclass(obj) and obj.__module__ is configs.__name__:
                print(name)
        exit(0)

    class_ = getattr(configs, args.env)
    env_config = class_()

    policy = models.PPOWrap(env_config.features, env_config.action_map, env_config.hidden)

    if args.reload is 'latest' or args.latest:
        _, args.reload = max(
            [(f.stat().st_mtime, f) for f in list(pathlib.Path('runs').glob(f'{env_config.gym_env_string}*/*.wgt'))])
    elif args.reload is 'config':
        args.reload = env_config.default_save[0]

    policy.load_state_dict(torch.load(args.reload))

    if env_config.players and env_config.players > 1:
        demo_adversarial(env_config, policy, args.speed, args.episodes)
    else:
        demo(env_config, policy, args.speed, args.episodes)
