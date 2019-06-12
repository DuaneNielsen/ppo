from models import PPOWrap, MultiPolicyNetContinuous, PPOWrapModel
import configs
from data import SARAdvantageDataset
from algos import *
import gym
from gym.wrappers import TimeLimit
import roboschool
from util import UniImageViewer, timeit
from rollout import single_episode
from data import Db
from statistics import mean
import tests.envs

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


class LineWalk(configs.GymDiscreteConfig):
    def __init__(self):
        super().__init__('LineWalk-v0')
        self.action_transform = OneHotDiscreteActionTransform(self.action_map)
        self.prepro = configs.NoPrePro()


def q_table(obs_size, action_size):
    states = torch.eye(obs_size).unsqueeze(1).expand(-1, action_size, -1).reshape(obs_size * action_size, obs_size)
    actions = torch.eye(action_size).unsqueeze(0).expand(obs_size, -1, -1).reshape(obs_size * action_size, action_size)
    return states, actions


class DummyQfunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.actions = 2
        self.x = nn.Linear(1, 1)

    def forward(self, state, action):
        lookup = torch.tensor([0.0, 1.0])
        lookup.unsqueeze(0).expand(action.size(0), -1)
        lookup.requires_grad = True
        return action.matmul(lookup)


def test_value():
    qf = DummyQfunc()
    print(qf(*q_table(4, 2)))


def test_policy(capsys):
    config = LineWalk()
    qfunc = DummyQfunc()
    policy = ValuePolicy(qfunc, EpsilonGreedyDiscreteDist, epsilon=0.00)
    exp_buffer = rollout_policy(1, policy, config, capsys)
    dataset = SARSDataset(exp_buffer, state_transform=DefaultTransform(), action_transform=config.action_transform)

    assert len(exp_buffer) == 4
    assert len(dataset) == 3

    state, action, reward, nxt, done = dataset[0]
    assert torch.allclose(state, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(action, torch.tensor([0.0, 1.0]))
    assert reward == 0.0
    assert torch.allclose(nxt, torch.tensor([0.0, 1.0, 0.0, 0.0]))

    state, action, reward, nxt, done = dataset[1]
    assert torch.allclose(state, torch.tensor([0.0, 1.0, 0.0, 0.0]))
    assert torch.allclose(nxt, torch.tensor([0.0, 0.0, 1.0, 0.0]))
    assert torch.allclose(action, torch.tensor([0.0, 1.0]))
    assert reward == 0.0

    state, action, reward, nxt, done = dataset[2]
    assert torch.allclose(state, torch.tensor([0.0, 0.0, 1.0, 0.0]))
    assert torch.allclose(nxt, torch.tensor([0.0, 0.0, 0.0, 1.0]))
    assert torch.allclose(action, torch.tensor([0.0, 1.0]))
    assert reward == 1.0


class Bandit(configs.GymDiscreteConfig):
    def __init__(self):
        super().__init__('Bandit-v0')
        self.action_transform = OneHotDiscreteActionTransform(self.action_map)
        self.prepro = configs.NoPrePro()


def test_policy_with_bandit(capsys):
    config = Bandit()
    qfunc = QMLP(config.features, len(config.action_map), config.features + len(config.action_map))
    policy = ValuePolicy(qfunc, EpsilonGreedyDiscreteDist, epsilon=0.1)
    exp_buffer = rollout_policy(1, policy, config, capsys)
    dataset = SARSDataset(exp_buffer, state_transform=DefaultTransform(), action_transform=config.action_transform)

    assert len(dataset) == 1
    state, action, reward, nxt = dataset[0]
    assert True


def test_one_step_td(capsys):
    config = Bandit()
    config.optimizer = 'SGD'
    config.lr = 0.1
    config.convergence_error = 1e-3
    config.wrappers.append(TimeLimit)
    #qfunc = QMLP(config.features, len(config.action_map), config.features + len(config.action_map))
    qfunc = QTable(config.features, len(config.action_map))
    one_step_td = OneStepTD(qfunc)
    policy = ValuePolicy(qfunc, EpsilonGreedyDiscreteDist, epsilon=0.3)

    states, actions = q_table(3, 2)
    values = policy.qf(states, actions)
    for i in range(2, 4):
        logger.info(f'{states[i]}, {actions[i]}, {values[i]}')

    for epoch in range(10):
        exp_buffer = rollout_policy(10, policy, config, capsys)
        policy = one_step_td(policy, exp_buffer, config)

        states, actions = q_table(3, 2)
        values = policy.qf(states, actions)
        test_policy = ValuePolicy(policy.qf, GreedyDiscreteDist)
        for i in range(2, 4):
            action = test_policy(states[i].unsqueeze(0)).sample()
            logger.info(f'{states[i]}, {actions[i]}, {values[i]}, {action}')
        logger.info(f'{qfunc.weights}')