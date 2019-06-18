import data.prepro
from configs import OptimizerConfig, PurePPOClipConfig, OneStepTDConfig, DataConfigItem
from data.transforms import DefaultTransform, OneHotDiscreteActionTransform, DiscreteActionTransform
import configs
from algos import *
from gym.wrappers import TimeLimit
from util import UniImageViewer
from rollout import single_episode
from data import Db
from statistics import mean
import tests.envs
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s-%(module)s-%(message)s', level=logging.INFO)


def rollout_policy(num_episodes, policy, config, capsys=None, render=False, render_freq=1, redis_host='localhost',
                   redis_port=6379):
    policy = policy.eval()
    policy = policy.to('cpu')
    db = Db(host=redis_host, port=redis_port, db=1)
    rollout = db.create_rollout(config.data.coder.construct())
    v = UniImageViewer(config.env.name, (200, 160))
    env = config.env.construct()
    rewards = []

    for i in range(num_episodes):
        render_iter = (i % render_freq == 0) and render
        episode = single_episode(env, config, policy, rollout, render=render_iter)
        rewards.append(episode.total_reward())

    rollout.finalize()
    logger.info(f'ave reward {mean(rewards)}')
    return rollout


def test_ppo_clip_discrete():
    config = configs.Discrete('LunarLander-v2')
    optimizer = OptimizerConfig(torch.optim.SGD, lr=0.1)
    config.algo = PurePPOClipConfig(optimizer, ppo_steps_per_batch=10)
    config.data.action_transform = DataConfigItem(DiscreteActionTransform, config.env.action_map)
    model = configs.ModelConfig(MultiPolicyNet, config.env.state_space_shape[0], config.env.actions,
                                hidden=config.env.state_space_shape[0])
    config.actor = model
    policy_net = PPOWrapModel(config.actor.construct())
    ppo = config.algo.construct()
    critic = None

    for epoch in range(3):
        exp_buffer = rollout_policy(10, policy_net, config)
        policy_net, critic = ppo(policy_net, None, exp_buffer, config)
        assert True


def test_ppo_clip_continuous(capsys):
    config = configs.Continuous('RoboschoolHalfCheetah-v1')
    policy_net = PPOWrapModel(config.actor.construct())
    algo = config.algo.construct()

    for epoch in range(3):
        exp_buffer = rollout_policy(10, policy_net, config, capsys)
        policy_net, critic = algo(policy_net, None, exp_buffer, config)


class LineWalk(configs.Discrete):
    def __init__(self):
        wrappers = [TimeLimit]
        super().__init__('LineWalk-v0', wrappers)
        self.data.action_transform = DataConfigItem(OneHotDiscreteActionTransform, self.env.action_map)
        self.data.prepro = DataConfigItem(data.prepro.NoPrePro)


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
    dataset = SARSDataset(exp_buffer, state_transform=DefaultTransform(), action_transform=config.data.action_transform)

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


class Bandit(configs.Discrete):
    def __init__(self):
        wrappers = [TimeLimit]
        super().__init__('Bandit-v0', wrappers)
        self.data.action_transform = DataConfigItem(OneHotDiscreteActionTransform, self.env.action_map)
        self.data.prepro = DataConfigItem(data.prepro.NoPrePro)


def test_policy_with_bandit(capsys):
    config = Bandit()
    qfunc = QMLP(config.env.state_space_shape[0], len(config.env.action_map),
                 config.env.state_space_shape[0] + len(config.env.action_map))
    policy = ValuePolicy(qfunc, EpsilonGreedyDiscreteDist, epsilon=0.1)
    exp_buffer = rollout_policy(1, policy, config, capsys)
    dataset = SARSDataset(exp_buffer, state_transform=DefaultTransform(), action_transform=config.data.action_transform.construct())

    assert len(dataset) == 1
    state, action, reward, nxt, done = dataset[0]
    assert True


def test_one_step_td(capsys):
    config = Bandit()
    optimizer = OptimizerConfig(torch.optim.SGD, lr=0.1)
    config.algo = OneStepTDConfig(optimizer)
    # qfunc = QMLP(config.features, len(config.action_map), config.features + len(config.action_map))
    qfunc = QTable(config.env.state_space_shape[0], len(config.env.action_map))
    one_step_td = OneStepTD()
    policy = ValuePolicy(qfunc, EpsilonGreedyDiscreteDist, epsilon=0.3)

    states, actions = q_table(3, 2)
    values = policy.qf(states, actions)
    for i in range(2, 4):
        logger.info(f'{states[i]}, {actions[i]}, {values[i]}')

    for epoch in range(10):
        exp_buffer = rollout_policy(10, policy, config, capsys)
        policy, qfunc = one_step_td(None, qfunc, exp_buffer, config)

        states, actions = q_table(3, 2)
        values = policy.qf(states, actions)
        test_policy = ValuePolicy(policy.qf, GreedyDiscreteDist)
        for i in range(2, 4):
            action = test_policy(states[i].unsqueeze(0)).sample()
            logger.info(f'{states[i]}, {actions[i]}, {values[i]}, {action}')
        logger.info(f'{qfunc.weights}')

    states, actions = q_table(3, 2)
    values = policy.qf(states, actions)
    assert (values[2].item() + 1.0) ** 2 < 0.01
    assert (values[3].item() - 1.0) ** 2 < 0.01


def test_one_step_td_linewalk(capsys):
    config = LineWalk()
    optimizer = OptimizerConfig(torch.optim.SGD, lr=0.1)
    config.algo = OneStepTDConfig(optimizer)
    qfunc = QTable(config.env.state_space_shape[0], len(config.env.action_map))
    one_step_td = config.algo.construct()

    policy = ValuePolicy(qfunc, EpsilonGreedyDiscreteDist, epsilon=0.3)
    states, actions = q_table(config.env.state_space_shape[0], config.env.actions)
    values = policy.qf(states, actions)
    for i in range(2, 4):
        logger.info(f'{states[i]}, {actions[i]}, {values[i]}')

    for epoch in range(30):
        exp_buffer = rollout_policy(100, policy, config, capsys)
        policy, qfunc = one_step_td(None, qfunc, exp_buffer, config)

        states, actions = q_table(config.env.state_space_shape[0], config.env.actions)
        values = policy.qf(states, actions)
        test_policy = ValuePolicy(policy.qf, GreedyDiscreteDist)
        for i in range(0, 6):
            action = test_policy(states[i].unsqueeze(0)).sample()
            logger.info(f'{states[i]}, {actions[i]}, {values[i]}, {action}')
        logger.info(f'{qfunc.weights}')


def test_one_step_td_LunarLander(capsys):
    config = configs.Discrete('LunarLander-v2')
    algo = config.algo.construct()
    actor = config.random_policy.construct()
    critic = config.critic.construct()

    for epoch in range(10):
        exp_buffer = rollout_policy(50, actor, config, capsys, render=True, render_freq=50)
        actor, critic = algo(None, critic, exp_buffer, config, device='cuda')
