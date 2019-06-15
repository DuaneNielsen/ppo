from data.coders import Step
from statistics import mean
from copy import copy

import logging
logger = logging.getLogger(__name__)


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

    episode = None
    if rollout is not None:
        episode = rollout.create_episode()
    episode_length = 0
    first_obs = env.reset()
    observation_t0 = copy(first_obs)
    observation_t1 = copy(first_obs)
    state = config.data.prepro(observation_t1, observation_t0)
    entropy = []

    done = False
    while not done:
        # take an action on current observation and record result
        state_t = config.data.transform(state, insert_batch=True)
        action_dist = policy(state_t)

        entropy.append(action_dist.entropy().mean().item())

        action = config.data.action_transform(action_dist.sample())

        observation_t1, reward, done, info = env.step(action)

        done = done or episode_length > config.gatherer.max_rollout_len

        if episode is not None:
            episode.append(Step(state, action, reward, False), config.gatherer.episode_batch_size)
        episode_length += 1

        # compute the observation that resulted from our action
        state = config.data.prepro(observation_t1, observation_t0)
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