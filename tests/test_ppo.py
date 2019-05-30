from models import PPOWrap, MultiPolicyNetContinuous, PPOWrapModel
import configs
from data import RolloutDatasetBase
from ppo_clip_discrete import train_policy, train_ppo_continuous
import gym
import roboschool
from util import UniImageViewer, timeit
from rollout import single_episode, single_episode_continous
from data import Db


@timeit
def rollout_policy(num_episodes, policy, config, capsys=None, redis_host='localhost', redis_port=6379):

    policy = policy.eval()
    policy = policy.to('cpu')
    db = Db(host=redis_host, port=redis_port)
    rollout = db.create_rollout(config)
    v = UniImageViewer(config.gym_env_string, (200, 160))
    env = gym.make(config.gym_env_string)

    for i in range(num_episodes):

        if config.continuous:
            episode = single_episode_continous(env, config, policy, rollout)
        else:
            episode = single_episode(env, config, policy, rollout)

        if capsys:
            with capsys.disabled():
                print(episode.total_reward())
        #config.tb.add_scalar('epi_len', len(episode), config.tb_step)
        #config.tb_step += 1

    #torch.save(policy.state_dict(), config.rundir + f'/latest.wgt')

    rollout.finalize()
    return rollout


def test_ppo_clip_discrete():

    config = configs.LunarLander()
    policy_net = PPOWrap(config.features, config.action_map, config.hidden)

    for epoch in range(3):

        rollout = rollout_policy(3, policy_net, config)
        dataset = RolloutDatasetBase(config, rollout)
        train_policy(policy_net, dataset, config)
        assert True


def test_ppo_clip_continuous(capsys):

    config = configs.HalfCheetah()
    model = MultiPolicyNetContinuous(config.model.features_size, config.model.action_size, config.model.hidden_size)
    policy_net = PPOWrapModel(model)

    for epoch in range(3):

        rollout = rollout_policy(10, policy_net, config, capsys)
        dataset = RolloutDatasetBase(config, rollout)
        train_ppo_continuous(policy_net, dataset, config)