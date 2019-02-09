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



if __name__ == '__main__':

    parser = argparse.ArgumentParser('learn to play pong')
    parser.add_argument('--env', default='LunarLander')
    parser.add_argument('--reload', default='latest')
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

    if args.reload is 'latest':
        _, args.reload = max([(f.stat().st_mtime, f) for f in list(pathlib.Path('runs').glob(f'{env_config.gym_env_string}*/*.wgt'))])

    policy.load_state_dict(torch.load(args.reload))

    demo(env_config, policy, args.speed)