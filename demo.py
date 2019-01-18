import torch
import argparse
import gym
from viewer import UniImageViewer
import cv2
#from vanilla_pong import PolicyNet
from ppo_clip_discrete import MultiPolicyNet
import ppo_clip_discrete
#from ppo_pong import PolicyNet as PPOPolicyNet
from torchvision.transforms.functional import to_tensor
import numpy as np
import time


def downsample(observation):
    greyscale = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    greyscale = cv2.resize(greyscale, downsample_image_size, cv2.INTER_LINEAR)
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

    downsample_image_size = (100, 80)
    features = downsample_image_size[0] * downsample_image_size[1]
    default_action = 2
    num_rollouts = 10

    # if args.multi:
    #     policy = MultiPolicyNet(features, [2, 3])
    # elif args.ppo:
    #     pong_action_map = [0, 2, 3]
    #     policy = PPOWrap(features, pong_action_map)
    # else:
    #     policy = PolicyNet(features)

    args.reload = r'runs\ppo_multilabel_752\vanilla.wgt'
    pong_action_map = [0, 2, 3]
    policy = ppo_clip_discrete.PPOWrap(features, pong_action_map)


    policy.load_state_dict(torch.load(args.reload))
    env = gym.make('Pong-v0')
    v = UniImageViewer('pong', (200, 160))

    if args.list:
        print(env.unwrapped.get_action_meanings())
        exit()

    for i in range(num_rollouts):

        game_length = 0
        gl = []

        observation_t0 = env.reset()
        observation_t0 = downsample(observation_t0)
        action = default_action
        observation_t1, reward, done, info = env.step(action)
        observation_t1 = downsample(observation_t1)
        observation = observation_t1 - observation_t0
        observation_t0 = observation_t1
        done = False

        while not done:
            observation_tensor = to_tensor(np.expand_dims(observation, axis=2)).squeeze().unsqueeze(0).view(-1,
                                                                                                            features)
            action_logprob = policy(observation_tensor)
            action = policy.new.max_action(action_logprob)
            # if action_prob > 0.9:
            #     action = 2
            # elif action_prob < 0.1:
            #     action = 3
            # else
            #     action = 0

            #action = 2 if np.random.uniform() < action_prob.item() else 3
            observation_t1, reward, done, info = env.step(action)

            env.render(mode='human')

            # compute the observation that resulted from our action
            observation_t1 = downsample(observation_t1)
            observation = observation_t1 - observation_t0
            observation_t0 = observation_t1

            time.sleep(args.speed)

