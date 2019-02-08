import torch
import argparse
import gym

import models
#from util import UniImageViewer
import ppo_clip_discrete
import time


if __name__ == '__main__':

    parser = argparse.ArgumentParser('learn to play pong')
    parser.add_argument('--env', default='LunarLander')
    parser.add_argument('--reload', default=r'runs\lander_118\vanilla.wgt')
    parser.add_argument('--speed', type=float, default=0.02)
    parser.set_defaults(multi=False, ppo=True)
    args = parser.parse_args()

    num_rollouts = 100

    class_ = getattr(ppo_clip_discrete, args.env)
    env_config = class_()

    policy = models.PPOWrap(env_config.features, env_config.action_map, env_config.hidden)

    print(args.reload)
    policy.load_state_dict(torch.load(args.reload))

    env = gym.make(env_config.gym_env_string)
    #v = UniImageViewer(env_config.gym_env_string, (200, 160))

    # if args.list:
    #     print(env.unwrapped.get_action_meanings())
    #     exit()

    for i in range(num_rollouts):

        episode_length = 0

        observation_t0 = env.reset()
        observation_t0 = env_config.prepro(observation_t0)
        action = env_config.default_action
        observation_t1, reward, done, info = env.step(action)
        observation_t1 = env_config.prepro(observation_t1)
        observation = observation_t1 - observation_t0
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
            observation_t1 = env_config.prepro(observation_t1)
            observation = observation_t1 - observation_t0
            observation_t0 = observation_t1

            time.sleep(args.speed)
            episode_length += 1

