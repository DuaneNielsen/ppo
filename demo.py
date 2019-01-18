import torch
import argparse
import gym

import models
from util import UniImageViewer
import cv2
#from vanilla_pong import PolicyNet
from models import MultiPolicyNet
import ppo_clip_discrete
from torchvision.transforms.functional import to_tensor
import numpy as np
import time


def prepro(observation):
    greyscale = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    greyscale = cv2.resize(greyscale, prepro_image_size, cv2.INTER_LINEAR)
    return greyscale


if __name__ == '__main__':

    parser = argparse.ArgumentParser('learn to play pong')
    parser.add_argument('--reload', default='vanilla691.wgt')
    parser.add_argument('--speed', type=float, default=0.02)
    parser.add_argument('--multi', dest='multi', action='store_true')
    parser.add_argument('--ppo', dest='ppo', action='store_true')
    parser.add_argument('--list-actions', dest='list', action='store_true')
    parser.set_defaults(multi=False, ppo=True)
    args = parser.parse_args()

    prepro_image_size = (100, 80)
    features = prepro_image_size[0] * prepro_image_size[1]
    default_action = 2
    num_rollouts = 10

    args.reload = r'runs\ppo_multilabel_807\vanilla.wgt'

    env_config = ppo_clip_discrete.LunarLander()

    pong_action_map = [0, 2, 3]
    policy = models.PPOWrap(env_config.features, env_config.action_map, env_config.hidden)


    policy.load_state_dict(torch.load(args.reload))
    env = gym.make(env_config.gym_env_string)
    v = UniImageViewer(env_config.gym_env_string, (200, 160))

    if args.list:
        print(env.unwrapped.get_action_meanings())
        exit()

    for i in range(num_rollouts):

        game_length = 0
        gl = []

        observation_t0 = env.reset()
        observation_t0 = env_config.prepro(observation_t0)
        action = default_action
        observation_t1, reward, done, info = env.step(action)
        observation_t1 = env_config.prepro(observation_t1)
        observation = observation_t1 - observation_t0
        observation_t0 = observation_t1
        done = False

        while not done:
            observation_tensor = env_config.transform(observation)
            action_logprob = policy(observation_tensor)
            action = policy.new.max_action(action_logprob)
            observation_t1, reward, done, info = env.step(action.squeeze().item())

            env.render(mode='human')

            # compute the observation that resulted from our action
            observation_t1 = env_config.prepro(observation_t1)
            observation = observation_t1 - observation_t0
            observation_t0 = observation_t1

            time.sleep(args.speed)

