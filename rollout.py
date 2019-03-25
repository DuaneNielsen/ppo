import gym
import torch

from data import Db, Step
from util import timeit, UniImageViewer, Init

# @timeit
# def rollout_adversarial_policy(policy, env_config, epoch):
#
#     policy = policy.eval()
#     policy = policy.to('cpu')
#     db = Db()
#     rollout = db.create_rollout(env_config)
#     reward_total = 0
#     env = gym.make(env_config.gym_env_string)
#
#     v1 = UniImageViewer(env_config.gym_env_string + ' player 1', (200, 160))
#     v2 = UniImageViewer(env_config.gym_env_string + ' player 2', (200, 160))
#
#     for i in range(num_rollouts):
#
#         episode_length = 0
#
#         observation_t0 = env.reset()
#         action = env_config.default_action
#         observation_t1, rewards, done, info = env.step((action, action))
#         observations = [env_config.prepro(t1, t0) for t1, t0 in zip(observation_t1, observation_t0)]
#         observation_t0 = observation_t1
#         done = False
#
#         while not done:
#             # take an action on current observation and record result
#             observation_tensors = [rollout.transform(o) for o in observations]
#             obs_stack = torch.stack(observation_tensors, dim=0)
#             action_prob = policy(obs_stack)
#             index, action = policy.sample(action_prob)
#             actions = [a.item() for a in action.chunk(action.size(0), dim=0)]
#             indices = [i.item() for i in index.chunk(index.size(0), dim=0)]
#
#             observation_t1, rewards, done, info = env.step(actions)
#
#             done = done or episode_length > env_config.max_rollout_len
#             reward_total += sum(rewards)
#             episode_length += 1
#
#             for i, (o, a, r) in enumerate(zip(observations, indices, rewards)):
#                 rollout.append(o, a, r, done, episode=i)
#
#             # compute the observation that resulted from our action
#             observations = [env_config.prepro(t1, t0) for t1, t0 in zip(observation_t1, observation_t0)]
#             observation_t0 = observation_t1
#
#             if view_games:
#                 v1.render(observations[0])
#                 v2.render(observations[1])
#                 env.render(mode='human')
#
#         # more monitoring
#         config.tb.add_scalar('reward', reward_total, config.tb_step)
#         reward_total = 0
#         config.tb.add_scalar('epi_len', episode_length, config.tb_step)
#         config.tb_step += 1
#
#     # save the file every so often
#     if epoch % 20 == 0:
#         torch.save(policy.state_dict(), config.rundir + '/vanilla.wgt')
#
#     rollout.end_rollout()
#     return rollout



@timeit
def rollout_policy(num_rollouts, policy, env_config, config):

    policy = policy.eval()
    policy = policy.to('cpu')
    db = Db()
    rollout = db.create_rollout(env_config)
    reward_total = 0

    v = UniImageViewer(env_config.gym_env_string, (200, 160))
    env = gym.make(env_config.gym_env_string)

    for i in range(num_rollouts):

        episode = rollout.create_episode()

        episode_length = 0

        observation_t0 = env.reset()
        action = env_config.default_action
        observation_t1, reward, done, info = env.step(action)
        observation = env_config.prepro(observation_t1, observation_t0)
        observation_t0 = observation_t1
        done = False

        while not done:
            # take an action on current observation and record result
            observation_tensor = env_config.transform(observation, insert_batch=True)
            action_prob = policy(observation_tensor)
            index, action = policy.sample(action_prob)

            observation_t1, reward, done, info = env.step(action.squeeze().item())

            done = done or episode_length > env_config.max_rollout_len
            reward_total += reward
            episode_length += 1

            episode.append(Step(observation, index, reward, done))

            # compute the observation that resulted from our action
            observation = env_config.prepro(observation_t1, observation_t0)
            observation_t0 = observation_t1

            if config.view_games:
                env.render(mode='human')
            if config.view_obs:
                v.render(observation)

        # more monitoring
        config.tb.add_scalar('reward', reward_total, config.tb_step)
        reward_total = 0

        config.tb.add_scalar('epi_len', episode_length, config.tb_step)
        config.tb_step += 1

    torch.save(policy.state_dict(), config.rundir + f'/latest.wgt')

    rollout.end()
    return rollout