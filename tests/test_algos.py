from models import PPOWrap, MultiPolicyNetContinuous, PPOWrapModel
import configs
from data import SARAdvantageDataset
from algos import *
import gym
import roboschool
from util import UniImageViewer, timeit
from rollout import single_episode
from data import Db
from statistics import mean

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s-%(module)s-%(message)s', level=logging.INFO)


@timeit
def rollout_policy(num_episodes, policy, config, capsys=None, redis_host='localhost', redis_port=6379):

    policy = policy.eval()
    policy = policy.to('cpu')
    db = Db(host=redis_host, port=redis_port, db=1)
    rollout = db.create_rollout(config)
    v = UniImageViewer(config.gym_env_string, (200, 160))
    env = gym.make(config.gym_env_string)
    if config.wrappers:
        for wrapper in config.wrappers:
            env = wrapper(env)
    rewards = []

    for i in range(num_episodes):

        episode = single_episode(env, config, policy, rollout)
        rewards.append(episode.total_reward())

    rollout.finalize()
    logger.info(f'ave reward {mean(rewards)}')
    return rollout


def test_ppo_clip_discrete():

    config = configs.LunarLander()
    model = config.model.get_model()
    policy_net = PPOWrapModel(model)
    ppo = PurePPOClip()

    for epoch in range(3):

        exp_buffer = rollout_policy(10, policy_net, config)
        policy_net = ppo(policy_net, exp_buffer, config)
        assert True


def test_ppo_clip_continuous(capsys):

    config = configs.HalfCheetah()
    model = MultiPolicyNetContinuous(config.model.features_size, config.model.action_size, config.model.hidden_size)
    policy_net = PPOWrapModel(model)
    ppo = PurePPOClip()

    for epoch in range(3):

        exp_buffer = rollout_policy(10, policy_net, config, capsys)
        policy_net = ppo(policy_net, exp_buffer, config)


def test_one_step_td(capsys):
    config = configs.FrozenLakeValue()
    qfunc = QMLP(config.features, len(config.action_map), config.features + len(config.action_map))
    one_step_td = OneStepTD(qfunc)
    policy = ValuePolicy(qfunc, EpsilonGreedyDiscreteDist, epsilon=0.05)

    for epoch in range(3):
        exp_buffer = rollout_policy(3, policy, config, capsys)
        policy = one_step_td(policy, exp_buffer, config)
