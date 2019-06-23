from data.coders import Step
from statistics import mean
from copy import copy

import logging
logger = logging.getLogger(__name__)

import pickle
import numpy as np
import torch
from models import EpsilonGreedyFlatDiscreteDist

# todo implement prepro as an env wrapper

def single_episode(env, config, policy, rollout=None, v=None, render=False, display_observation=False):

    """

    :param config: The General configuration
    :param env: The simulator, reset will be called at the start of each episode
    :param config: the simulator configuration
    :param policy: the policy to run
    :param rollout: the rollout data structure to capture experience into
    :param v: an object with a render method for displaying images
    :param render: if True, env.render will be called for each step
    :param display_observation: if true, v.render(observation) will be called for each step
    :return:
    """

    prepro = config.data.prepro.construct()
    transform = config.data.transform.construct()
    action_transform = config.data.action_transform.construct()


    episode = None
    if rollout is not None:
        episode = rollout.create_episode()
    episode_length = 0
    first_obs = env.reset()
    observation_t0 = copy(first_obs)
    observation_t1 = copy(first_obs)
    state = prepro(observation_t1, observation_t0)
    entropy = []

    done = False
    while not done:
        # take an action on current observation and record result
        state_t = transform(state, insert_batch=True)
        action_dist = policy(state_t)

        entropy.append(action_dist.entropy().mean().item())

        action = action_transform(action_dist.sample())

        observation_t1, reward, done, info = env.step(action)

        done = done or episode_length > config.gatherer.max_rollout_len

        if episode is not None:
            episode.append(Step(state, action, reward, False), config.gatherer.episode_batch_size)
        episode_length += 1

        # compute the observation that resulted from our action
        state = prepro(observation_t1, observation_t0)
        observation_t0 = observation_t1

        if render:
            env.render(mode='human')
        if display_observation:
            v.render(state)

    if episode is not None:
        episode.append(Step(state, config.env.default_action, 0.0, True))
        episode.end()
        episode.entropy = mean(entropy)
    return episode


def single_discrete_episode_with_lookahead(env, config, policy, rollout=None, v=None, render=False, display_observation=False):

    """

    :param config: The General configuration
    :param env: The simulator, reset will be called at the start of each episode
    :param config: the simulator configuration
    :param policy: the policy to run
    :param rollout: the rollout data structure to capture experience into
    :param v: an object with a render method for displaying images
    :param render: if True, env.render will be called for each step
    :param display_observation: if true, v.render(observation) will be called for each step
    :return:
    """

    prepro = config.data.prepro.construct()
    transform = config.data.transform.construct()
    action_transform = config.data.action_transform.construct()

    episode = None
    if rollout is not None:
        episode = rollout.create_episode()
    episode_length = 0
    first_obs = env.reset()
    observation_t0 = copy(first_obs)
    observation_t1 = copy(first_obs)
    state = prepro(observation_t1, observation_t0)
    entropy = []

    done = False
    while not done:
        prev = pickle.dumps(env)
        value = np.zeros(len(config.env.action_map))
        for i, action in enumerate(config.env.action_map):
            observation_t1, reward, done, info = env.step(action)
            peek_state = prepro(observation_t1, observation_t0)
            state_t = transform(peek_state, insert_batch=True)
            value[i] = policy(state_t).item() + reward
            env = pickle.loads(prev)

        # take an action on current observation and record result
        # the below is argmax, you could also put a probability distribution here
        #action = torch.tensor([np.argmax(value)])

        action = EpsilonGreedyFlatDiscreteDist(torch.softmax(torch.from_numpy(value), dim=0), epsilon=0.05).sample().unsqueeze(0)

        action = action_transform(action)

        observation_t1, reward, done, info = env.step(action)

        done = done or episode_length > config.gatherer.max_rollout_len

        if episode is not None:
            episode.append(Step(state, action, reward, False), config.gatherer.episode_batch_size)
        episode_length += 1

        # compute the observation that resulted from our action
        state = prepro(observation_t1, observation_t0)
        observation_t0 = observation_t1

        if render:
            env.render(mode='human')
        if display_observation:
            v.render(state)

    if episode is not None:
        episode.append(Step(state, config.env.default_action, 0.0, True))
        episode.end()
        #episode.entropy = mean(entropy)
    return episode