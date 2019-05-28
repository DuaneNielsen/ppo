from models import PPOWrap
import configs
from rollout import rollout_policy
from data import RolloutDatasetBase
from ppo_clip_discrete import train_policy


def test_ppo_clip_discrete():

    config = configs.LunarLander()
    policy_net = PPOWrap(config.features, config.action_map, config.hidden)

    for epoch in range(3):

        rollout = rollout_policy(config.num_rollouts, policy_net, config)
        dataset = RolloutDatasetBase(config, rollout)
        train_policy(policy_net, dataset, config)
        assert True


def test_ppo_clip_continous():

    config = configs.LunarLander()
    policy_net = PPOWrap(config.features, config.action_map, config.hidden)

    for epoch in range(3):

        rollout = rollout_policy(config.num_rollouts, policy_net, config)
        dataset = RolloutDatasetBase(config, rollout)
        train_policy(policy_net, dataset, config)
        assert True