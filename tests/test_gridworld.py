import gym
from data.transforms import *
import gym_duane
import numpy as np
from statistics import mean
from torch import nn
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import logging
from models import GreedyDist
from colorama import Style, Fore
from collections import deque
from util import Timer
import objgraph


logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s-%(module)s-%(message)s', level=logging.DEBUG)
timer = Timer()


class BatchStep:
    def __init__(self, state, action, reward, done, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done
        self.next_state = next_state

    def __getitem__(self, item):
        return Step(self.state[item], self.action[item], self.reward[item], self.done[item], self.next_state[item])


class Step:
    def __init__(self, state, action, reward, done, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done
        self.next_state = next_state


class RewardAccumulator:
    def __init__(self, n, device='cpu'):
        self.accum = torch.zeros(n, requires_grad=False, device=device)
        self.reward_total = []

    def add(self, reward, done):
        with torch.no_grad():
            self.accum += reward
            self.reward_total.append(self.accum[done])
            self.accum[done] = 0.0

    def ave_reward(self):
        with torch.no_grad():
            r = torch.cat(self.reward_total)
            return torch.mean(r).item(), r.size(0)


class RewardAccumulatorNumpy:
    def __init__(self, n):
        self.n = n
        self.accum = np.zeros(n)
        self.episode_reward = np.array([])

    def add(self, reward, done):
        self.accum += reward
        d = done.astype(np.bool)
        self.episode_reward = np.append(self.episode_reward, self.accum[d], axis=0)
        self.accum[d] = 0.0

    def reset(self):
        self.accum = np.zeros(self.n)
        self.episode_reward = np.array([])


class RunningReward:
    def __init__(self, n, depth):
        self.n = n
        self.accum = np.zeros(n)
        self.recent = deque([], maxlen=depth)

    def add(self, reward, done):
        self.accum += reward
        d = done.astype(np.bool)
        self.recent.extend(self.accum[d].tolist())
        self.accum[d] = 0.0

    def reset(self):
        self.accum = np.zeros(self.n)
        self.recent.clear()

    def log(self):
        if len(self.recent) > 0:
            logger.info(f'{Fore.GREEN} reward {mean(self.recent)} {Style.RESET_ALL} epi {len(self.recent)}')
        else:
            logger.info(f'reward 0 epi {len(self.recent)}')


def batch_episode(env, policy, device, max_rollout_len=4000, v=None, render=False, display_observation=False):
    episode = []
    entropy = []

    state = env.reset()
    rwa = RewardAccumulator(state.size(0), device)

    if render:
        env.render()

    for _ in range(max_rollout_len):

        action_dist = policy(state)

        entropy.append(action_dist.entropy().mean().item())

        action = action_dist.sample()

        next_state, reward, done, info = env.step(action)

        episode.append(BatchStep(state.cpu().numpy(), action.cpu().numpy(), reward.cpu().numpy(), done.cpu().numpy(),
                            next_state.cpu().numpy()))

        rwa.add(reward, done)

        if render:
            env.render(mode='human')
        if display_observation:
            v.render(state)

        state = next_state

    final_entropy = mean(entropy)
    ave_reward, episodes = rwa.ave_reward()
    return episode, final_entropy, ave_reward, episodes


def one_step(env, state, policy, episode, v=None, render=False, display_observation=False):

    action_dist = policy(state)

    action = action_dist.sample()

    next_state, reward, done, info = env.step(action)

    episode.append(BatchStep(state.cpu().numpy(), action.cpu().numpy(), reward.cpu().numpy(), done.cpu().numpy(),
                        next_state.cpu().numpy()))

    if render:
        env.render(mode='human')
    if display_observation:
        v.render(state)

    return next_state, reward, done


class RandomPolicy:
    def __call__(self, state):
        p = torch.ones(state.size(0), 4) / 4
        return Categorical(p)


class EpsilonGreedyProperDiscreteDist:
    def __init__(self, probs, epsilon=0.05):
        self.probs = probs
        self.epsilon = epsilon

        e = self.epsilon / (self.probs.size(1) - 1)
        max = torch.argmax(self.probs, dim=1)
        self.p = torch.ones_like(self.probs, device=probs.device) * e
        self.p[torch.arange(self.p.size(0)), max] = 1.0 - self.epsilon

    def sample(self):
        return Categorical(self.p).sample()

    def entropy(self):
        return torch.sum(- self.probs * torch.log2(self.probs), dim=1)

    def logprob(self, action):
        probs = torch.sum(self.p * action, dim=1)
        return torch.log(probs)


class QPolicy(nn.Module):
    def __init__(self, qf, actions, dist_class, **kwargs):
        super().__init__()
        self.qf = qf
        self.actions = torch.arange(actions)
        self.num_actions = actions
        self.dist_class = dist_class
        self.kwargs = kwargs

    def parameters(self, recurse=True):
        return self.qf.parameters(recurse)

    def forward(self, state):
        batch_size = state.size(0)
        input_size = state.shape[1:]

        # copy the states * number of actions
        states = state.unsqueeze(1).expand(batch_size, self.num_actions, *input_size)
        states = states.reshape(batch_size * self.num_actions, *input_size)

        # repeat the actions for each state in the batch
        actions = self.actions.unsqueeze(0).expand(batch_size, -1)
        actions = actions.reshape(batch_size * self.num_actions)

        values = self.qf(states, actions)
        values = values.reshape(batch_size, self.num_actions)

        probs = torch.softmax(values, dim=1)

        return self.dist_class(probs, **self.kwargs)


def gradnorm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


class DiscreteQTable(nn.Module):
    def __init__(self, features, actions):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(actions, *features))

    def forward(self, state, action):
        return torch.sum(self.weights[action, :, :] * state, dim=[1, 2])


class SARSGridDataset(Dataset):
    def __init__(self, episode):
        super().__init__()
        self.episode = episode
        self.batch_size = episode[0].state.shape[0]

    def _transform(self, step, resetting):
        state = torch.from_numpy(step.state)
        action = torch.tensor(step.action.item())
        reward = torch.tensor(step.reward.item())
        done = torch.tensor(step.done.item(), dtype=torch.uint8)
        resetting = torch.tensor(resetting, dtype=torch.uint8)
        next_state = torch.from_numpy(step.next_state)
        return state, action, reward, done, resetting, next_state

    def __getitem__(self, item):
        t = item // self.batch_size
        offset = item % self.batch_size
        step = self.episode[t][offset]
        if t > 0:
            resetting = self.episode[t - 1][offset].done.item()
        else:
            resetting = 0
        return self._transform(step, resetting)

    def __len__(self):
        return len(self.episode) * self.batch_size


class SARSGridDataLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        sample = np.random.randint(len(self.dataset), size=self.batch_size)
        data = []
        for item in np.nditer(sample):
            data.append(self.dataset[item])
        batch = default_collate(data)
        return batch


def test_SARSGridDataset():
    state = np.array([
        [[1, 0, 0]],
        [[1, 0, 0]]
    ])
    action = np.array([
        [0],
        [0]
    ])
    reward = np.array([
        [0.0],
        [0.0]
    ])
    done = np.array([
        [1],
        [1]
    ])
    next_state = np.array([
        [[0, 1, 0]],
        [[0, 1, 0]]
    ])

    a = BatchStep(state, action, reward, done, next_state)

    state = np.array([
        [[0, 1, 0]],
        [[0, 1, 0]]
    ])
    action = np.array([
        [0],
        [0]
    ])
    reward = np.array([
        [0.0],
        [0.0]
    ])
    done = np.array([
        [0],
        [0]
    ])
    next_state = np.array([
        [[1, 0, 0]],
        [[1, 0, 0]]
    ])

    b = BatchStep(state, action, reward, done, next_state)

    episode = [a, b]

    dataset = SARSGridDataset(episode)

    state, action, reward, done, reset, next_state = dataset[0]
    print(state, action, reward, done, reset)

    state, action, reward, done, reset, next_state = dataset[1]
    print(state, action, reward, done, reset)

    state, action, reward, done, reset, next_state = dataset[2]
    print(state, action, reward, done, reset)

    state, action, reward, done, reset, next_state = dataset[3]
    print(state, action, reward, done, reset)


def test_reset():
    env = gym.make('SimpleGrid-v2', n=1, map_string="""
    [
    [T(-1.0), S, T(1.0)]
    ]
    """)

    device = 'cpu'

    episode = []

    action = torch.tensor([0])

    state = env.reset()
    next_state, reward, done, info = env.step(action)

    episode.append(BatchStep(state.numpy(), action.numpy(), reward.numpy(), done.numpy(), next_state.numpy()))

    state = next_state
    next_state, reward, done, info = env.step(action)

    episode.append(BatchStep(state.numpy(), action.numpy(), reward.numpy(), done.numpy(), next_state.numpy()))

    dataset = SARSGridDataset(episode=episode)

    state, action, reward, done, reset, next_state = dataset[1]

    print(state)
    print(next_state)

    assert reset == 1


def train(episode, critic, device, optim, actions, discount_factor=0.99, epsilon=0.05, logging_freq=10, batch_size=10000, num_workers=12):
    dataset = SARSGridDataset(episode)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

    greedy_policy = QPolicy(critic, actions=actions, dist_class=GreedyDist)

    for state, action, reward, done, reset, next_state in loader:
        state = state.to(device)
        action = action.to(device)
        reward = reward.to(device)
        done = done.to(device)
        reset = reset.to(device)
        next_state = next_state.to(device)

        # remove transtions going from Terminal => Start
        state = state[~reset]
        action = action[~reset]
        reward = reward[~reset]
        done = done[~reset]
        next_state = next_state[~reset]

        # zero the boostrapped value of terminal states
        zero_if_terminal = (~done).to(next_state.dtype)

        loss = torch.tensor([100.0])
        prev_loss = torch.tensor([101.0])
        i = 0

        while loss.item() > 0.01 and abs(loss.item() - prev_loss.item()) > 0.0001:
            # for _ in range(1):
            i += 1
            prev_loss = loss

            # softmax and lack of logprob will affect the calculation here!
            next_action = greedy_policy(next_state).sample().to(device)

            next_value = critic(next_state, next_action)
            target = reward + zero_if_terminal * discount_factor * next_value

            optim.zero_grad()
            predicted = critic(state, action)
            error = (target - predicted)
            loss = torch.mean(error ** 2)
            loss.backward()
            optim.step()

            if logging_freq > 0 and i % logging_freq == 0:
                log_stats(action, critic, dataset, i, loss, predicted, state, target)

        if logging_freq > 0:
            log_stats(action, critic, dataset, i, loss, predicted, state, target)
        logger.info(f'iterations {i}')
    # return an epsilon greedy policy as actor
    return QPolicy(critic, actions=actions, dist_class=EpsilonGreedyProperDiscreteDist, epsilon=epsilon), critic


def train_one(episode, critic, device, optim, actions, discount_factor=0.99, epsilon=0.05, logging_freq=10, batch_size=10000, num_workers=12):
    dataset = SARSGridDataset(episode)
    loader = SARSGridDataLoader(dataset, batch_size=batch_size)

    greedy_policy = QPolicy(critic, actions=actions, dist_class=GreedyDist)

    for state, action, reward, done, reset, next_state in loader:

        state = state.to(device)
        action = action.to(device)
        reward = reward.to(device)
        done = done.to(device)
        reset = reset.to(device)
        next_state = next_state.to(device)

        # remove transtions going from Terminal => Start
        state = state[~reset]
        action = action[~reset]
        reward = reward[~reset]
        done = done[~reset]
        next_state = next_state[~reset]

        # zero the boostrapped value of terminal states
        zero_if_terminal = (~done).to(next_state.dtype)

        # softmax and lack of logprob will affect the calculation here!
        next_action = greedy_policy(next_state).sample().to(device)

        next_value = critic(next_state, next_action)
        target = reward + zero_if_terminal * discount_factor * next_value

        optim.zero_grad()
        predicted = critic(state, action)
        error = (target - predicted)
        loss = torch.mean(error ** 2)
        loss.backward()
        optim.step()

        break

    # return an epsilon greedy policy as actor
    return QPolicy(critic, actions=actions, dist_class=EpsilonGreedyProperDiscreteDist, epsilon=epsilon), critic


def log_stats(action, critic, dataset, i, loss, predicted, state, target):
    with torch.no_grad():
        current = critic(state, action)
        total_diff = torch.abs(predicted - current).sum().item()
        mean_diff = total_diff / len(dataset)
        magnitude = gradnorm(critic)
    logger.info(f'loss {loss.item()}')
    logger.info(f'grdnrm {magnitude}')
    logger.info(f'mean_dif {mean_diff}')
    logger.info(
        f'prev mean {predicted.mean()} std {predicted.std()} max {predicted.max()} min {predicted.min()}')
    logger.info(f'target mean {target.mean()} std {target.std()} max {target.max()} min {target.min()}')
    logger.info(
        f'current mean {current.mean()} std {current.std()} max {current.max()} min {current.min()}')
    logger.info(f'iterations {i}')


def test_bandit():
    device = 'cuda'
    ll_runs = 1000
    env = gym.make('SimpleGrid-v2', n=ll_runs, device=device, map_string="""
    [
    [T(-1.0), S, T(1.0)]
    ]
    """)


    obs = env.reset()
    critic = DiscreteQTable((env.height, env.width), 2).to(device)
    critic.weights.data[0, 0, 1] = 1.0
    critic.weights.data[1, 0, 1] = -1.0
    optim = torch.optim.SGD(critic.parameters(), lr=0.1)
    policy = QPolicy(critic, 2, EpsilonGreedyProperDiscreteDist, epsilon=0.05).to(device)
    ave_reward = 0

    while ave_reward < 0.95:
        print(f'paramters: {critic.weights.data}')
        episode, entropy, ave_reward, episodes = batch_episode(env, policy, device, max_rollout_len=50, render=False)
        logger.info(f'{Fore.GREEN}ave_reward {ave_reward} episodes {episodes} {Style.RESET_ALL}')
        policy, critic = train(episode, critic, device, optim, actions=2, epsilon=0.05, logging_freq=0)


def test_bandit_deepq():
    device = 'cuda'
    ll_runs = 1000
    env = gym.make('SimpleGrid-v2', n=ll_runs, device=device, map_string="""
    [
    [T(-1.0), S, T(1.0)]
    ]
    """)

    critic = DiscreteQTable((env.height, env.width), 2).to(device)
    critic.weights.data[0, 0, 1] = 1.0
    critic.weights.data[1, 0, 1] = -1.0
    optim = torch.optim.SGD(critic.parameters(), lr=0.1)
    policy = QPolicy(critic, 2, EpsilonGreedyProperDiscreteDist, epsilon=0.05).to(device)
    rw = RunningReward(ll_runs, 2000)
    ave_reward = 0
    exp_buffer = []

    state = env.reset()

    while ave_reward < 0.95:

        state, reward, done = one_step(env, state, policy, exp_buffer)
        rw.add(reward.cpu().numpy(), done.cpu().numpy())
        logger.info(f'reward {mean(rw.recent)} epi {len(rw.recent)}')
        policy, critic = train_one(exp_buffer, critic, device, optim, actions=2, epsilon=0.05, logging_freq=0)
        if len(exp_buffer) > 100:
            exp_buffer.pop(0)

        print_qvalues(critic.weights.data)


def test_shortwalk_deepq():
    device = 'cuda'
    ll_runs = 8000
    actions = 4
    env = gym.make('SimpleGrid-v2', n=ll_runs, device=device, map_string="""
    [
    [S, E, E],
    [E, E, T(1.0)]
    ]
    """)

    critic = DiscreteQTable((env.height, env.width), actions).to(device)
    critic.weights.data[0, 0, 1] = 1.0
    critic.weights.data[1, 0, 1] = -1.0
    optim = torch.optim.SGD(critic.parameters(), lr=0.1)
    policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=0.1).to(device)
    rw = RunningReward(ll_runs, 2000)
    ave_reward = 0
    exp_buffer = []

    state = env.reset()

    while ave_reward < 0.95:

        state, reward, done = one_step(env, state, policy, exp_buffer, render=True)
        rw.add(reward.cpu().numpy(), done.cpu().numpy())
        if len(rw.recent) > 0:
            logger.info(f'reward {mean(rw.recent)} epi {len(rw.recent)}')
        else:
            logger.info(f'reward 0 epi {len(rw.recent)}')
        rw.reset()

        policy, critic = train_one(exp_buffer, critic, device, optim, actions=actions, epsilon=0.1, logging_freq=0)
        if len(exp_buffer) > 100:
            exp_buffer.pop(0)

        print_qvalues(critic.weights.data)


def run_deep_q_on(map, ll_runs, replay_window=1000, epsilon=0.06, batch_size=10000, workers=12, logging_freq=10):

    device = 'cuda'
    actions = 4
    env = gym.make('SimpleGrid-v2', n=ll_runs, device=device, map_string=map)

    critic = DiscreteQTable((env.height, env.width), actions).to(device)
    critic.weights.data[0, 0, 1] = 1.0
    critic.weights.data[1, 0, 1] = -1.0
    optim = torch.optim.SGD(critic.parameters(), lr=0.1)
    policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=epsilon).to(device)
    rw = RunningReward(ll_runs, 2000)
    ave_reward = 0
    exp_buffer = deque(maxlen=replay_window)
    i = 0

    state = env.reset()

    while ave_reward < 0.95:
        if i % logging_freq == 0:
            logger.info(f"{Fore.LIGHTBLUE_EX}exp buffer: {len(exp_buffer)}{Style.RESET_ALL}")
            timer.start('main_loop')
            timer.start('step')
        state, reward, done = one_step(env, state, policy, exp_buffer, render=i % logging_freq == 0)
        if i % logging_freq == 0:
            timer.elapsed('step')

        if i % logging_freq == 0:
            rw.add(reward.cpu().numpy(), done.cpu().numpy())
            rw.log()
            rw.reset()

            timer.start('train_one')
        policy, critic = train_one(exp_buffer, critic, device, optim, actions=actions, epsilon=epsilon,
                                   batch_size=batch_size, logging_freq=0, num_workers=workers)
        if i % logging_freq == 0:
            timer.elapsed('train_one')
            print_qvalues(critic.weights.data)
            timer.elapsed('main_loop')

        i += 1


def print_qvalues(weights):
    ms = torch.argmax(weights, dim=0)
    arrows = [u'\N{BLACK LEFT-POINTING TRIANGLE}', u'\N{BLACK RIGHT-POINTING TRIANGLE}', u'\N{BLACK UP-POINTING TRIANGLE}', u'\N{BLACK DOWN-POINTING TRIANGLE}']

    for i in range(ms.size(0)):
        s = ''
        for j in range(ms.size(1)):
            s = s + arrows[ms[i, j].item()]
        logger.info(s)


def test_frozenlake():
    map = """
    [
    [T, T, T, T, T],
    [S, E, E, E, E],
    [E, E, E, E, E],
    [E, E, T, E, T],
    [E, E, E, E, E],
    [E, E, T, E, T(1.0)]
    ]
    """
    run_deep_q_on(map, ll_runs=8000, epsilon=0.1, replay_window=20, batch_size=10000, workers=12)


def test_fake_lunar_lander():
    map = """
    [
    [T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0)],
    [T(-1.0), E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, T(-1.0)],
    [T(-1.0), E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, T(-1.0)],
    [T(-1.0), E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, T(-1.0)],
    [T(-1.0), E, E, E, S, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, T(1.0)],
    [T(-1.0), E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, T(-1.0)],
    [T(-1.0), E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, T(-1.0)],
    [T(-1.0), E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, T(-1.0)],
    [T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0)]
    ]
    """
    run_deep_q_on(map, ll_runs=8000, epsilon=0.1, replay_window=100, batch_size=16000, workers=12)


def test_grid_walk():
    device = 'cuda'
    ll_runs = 8000
    actions = 4
    env = gym.make('SimpleGrid-v2', n=ll_runs, device=device, map_string="""
    [
    [S, E, E],
    [E, E, T(1.0)]
    ]
    """)


    obs = env.reset()
    critic = DiscreteQTable((env.height, env.width), actions).to(device)
    optim = torch.optim.SGD(critic.parameters(), lr=0.1)
    policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=0.05).to(device)
    ave_reward = 0
    episodes = 0

    while episodes < ll_runs:
        print(f'paramters: {critic.weights.data}')
        print_qvalues(critic.weights.data)
        episode, entropy, ave_reward, episodes = batch_episode(env, policy, device, max_rollout_len=4, render=False)
        logger.info(f'{Fore.GREEN}ave_reward {ave_reward} episodes {episodes} {Style.RESET_ALL}')
        policy, critic = train(episode, critic, device, optim, actions=actions, epsilon=0.05, logging_freq=0)