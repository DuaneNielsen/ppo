from __future__ import print_function
import os


from util import Converged
from torch.utils.data import DataLoader
import logging
gpu_profile = True
if gpu_profile:
    from util import gpu_profile
    import sys
    #sys.settrace(gpu_profile)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['GPU_DEBUG'] = '0'

logger = logging.getLogger(__name__)
from time import time
from data.data import SARAdvantageDataset, SARSDataset
from models import *
import numpy as np
import gc

# default optimizers


def ppo_loss(newprob, oldprob, advantage, clip=0.2):
    ratio = newprob / oldprob

    clipped_ratio = ratio.clamp(1.0 - clip, 1.0 + clip)
    clipped_step = clipped_ratio * advantage
    full_step = ratio * advantage
    min_step = torch.stack((full_step, clipped_step), dim=1)
    min_step, clipped = torch.min(min_step, dim=1)

    logger.debug(f'ADVTG {advantage[0].data}')
    logger.debug(f'NEW_P {newprob[0].data}')
    logger.debug(f'OLD_P {oldprob[0].data}')
    logger.debug(f'RATIO {ratio[0].data}')
    logger.debug(f'CLIP_ {clipped_step[0].data}')

    min_step *= -1
    return min_step.mean()


def ppo_loss_log(newlogprob, oldlogprob, advantage, clip=0.2):
    log_ratio = (newlogprob - oldlogprob)
    # clamp the log to stop infinities (85 is for 32 bit floats)
    log_ratio.clamp_(min=-10.0, max=10.0)
    ratio = torch.exp(log_ratio)

    clipped_ratio = ratio.clamp(1.0 - clip, 1.0 + clip)
    clipped_step = clipped_ratio * advantage.unsqueeze(1)
    full_step = ratio * advantage.unsqueeze(1)
    min_step = torch.stack((full_step, clipped_step), dim=1)
    min_step, clipped = torch.min(min_step, dim=1)

    # logger.info(f'mean advantage : {advantage.mean()}')
    # logger.info(f'mean newlog    : {newlogprob.mean()}')
    # logger.info(f'mean oldlob    : {oldlogprob.mean()}')
    # logger.info(f'mean log_ratio : {log_ratio.mean()}')
    # logger.info(f'mean ratio     : {ratio.mean()}')
    # logger.info(f'mean clip ratio: {clipped_ratio.mean()}')
    # logger.info(f'mean clip step : {clipped_step.mean()}')

    min_step *= -1
    return min_step.mean()





class PurePPOClip:
    def __call__(self, actor, critic, exp_buffer, config, device='cpu'):

        transform, action_transform = config.data.transforms()

        optim = config.algo.optimizer.construct(actor.parameters())
        actor = actor.train()
        actor = actor.to(device)
        dataset = SARAdvantageDataset(exp_buffer, discount_factor=config.algo.discount_factor,
                                      state_transform=transform,
                                      action_transform=action_transform, precision=config.data.precision)
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

        batches_p = 0
        for observation, action, reward, advantage in loader:
            batches_p += 1
            for step in range(config.algo.ppo_steps_per_batch):

                #todo catergorical distrubution loss.backward() super slow (pytorch problem)
                optim.zero_grad()

                new_dist = actor(observation)
                new_logprob = new_dist.log_prob(action)
                new_logprob.retain_grad()

                old_dist = actor(observation, old=True)
                old_logprob = old_dist.log_prob(action)
                actor.backup()

                loss = ppo_loss_log(new_logprob, old_logprob, advantage, clip=0.2)
                #logging.info(f'loss {loss.item()}')

                starttime = time()
                loss.backward()
                endtime = time()

                optim.step()

                logger.debug(endtime - starttime)

        logging.info(f'processed {batches_p} batches')
        if config.gpu_profile:
            gpu_profile(frame=sys._getframe(), event='line', arg=None)

        return actor, None


class OneStepTD:
    def __call__(self, actor, critic, exp_buffer, config, device='cpu'):

        transform, action_transform = config.data.transforms()

        one_step_config = config.algo
        critic = critic.to(device)
        greedy_policy = ValuePolicy(critic, GreedyDiscreteOneHotDist).to(device)
        pin_memory = device == 'cuda'
        optim = one_step_config.optimizer.construct(critic.parameters())
        dataset = SARSDataset(exp_buffer, state_transform=transform, action_transform=action_transform)
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True, pin_memory=pin_memory)
        c = Converged(one_step_config.min_change, detections=one_step_config.detections,
                      detection_window=one_step_config.detection_window)

        loss = torch.tensor([-1.0])
        iter = 0

        for state, action, reward, next_state, done in loader:

            state = state.to(device)
            action = action.to(device)
            reward = reward.to(device)
            next_state = next_state.to(device)
            done = done.to(device)

            while not c.converged(loss.item()):
                iter += 1

                zero_if_terminal = (~done).to(next_state.dtype)
                next_action = greedy_policy(next_state).sample().to(device)
                next_value = critic(next_state, next_action)
                target = reward + zero_if_terminal * config.algo.discount_factor * next_value

                optim.zero_grad()
                predicted = critic(state, action)
                error = target - predicted
                loss = torch.mean(error ** 2)
                loss.backward()
                optim.step()

                if iter % one_step_config.logging_freq == 0:
                    max_error = error.abs().max()
                    min_error = error.abs().min()
                    mean_error = error.abs().mean()
                    logger.info(f'loss     : {loss.item()}')
                    logger.info(f'max_error: {max_error.item()}')
                    logger.info(f'min_error: {min_error.item()}')
                    logger.info(f'mean_error: {mean_error.item()}')

        # return an epsilon greedy policy as actor
        return ValuePolicy(critic, EpsilonGreedyDiscreteDist, epsilon=one_step_config.epsilon), critic


class BootstrapValue:
    def __call__(self, actor, critic, exp_buffer, config, device='cpu'):
        transform, action_transform = config.data.transforms()

        algo = config.algo
        critic = critic.to(device)
        pin_memory = device == 'cuda'
        optim = algo.optimizer.construct(critic.parameters())
        dataset = SARSDataset(exp_buffer, state_transform=transform, action_transform=action_transform)
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True, pin_memory=pin_memory)
        c = Converged(algo.min_change, detections=algo.detections,
                      detection_window=algo.detection_window)

        loss = torch.tensor([-1.0])

        for state, action, reward, next_state, done in loader:

            state = state.to(device)
            reward = reward.to(device)
            next_state = next_state.to(device)
            done = done.to(device)
            prev = None
            current = None
            iter = 0

            logger.info(torch.sum(state, dim=0))

            while not c.converged(current, prev):
                iter += 1

                zero_if_terminal = (~done).to(next_state.dtype)
                next_value = critic(next_state)
                target = reward + zero_if_terminal * config.algo.discount_factor * next_value

                optim.zero_grad()
                predicted = critic(state)
                error = target - predicted
                loss = torch.mean(error ** 2)
                loss.backward()
                optim.step()

                with torch.no_grad():
                    prev = predicted.detach()
                    current = critic(state).detach()

                if iter % algo.logging_freq == 0:
                    max_error = error.abs().max()
                    min_error = error.abs().min()
                    mean_error = error.abs().mean()
                    logger.info(f'iterations: {iter}')
                    logger.info(f'loss     : {loss.item()}')
                    logger.info(f'max_error: {max_error.item()}')
                    logger.info(f'min_error: {min_error.item()}')
                    logger.info(f'mean_error: {mean_error.item()}')

        return critic, critic


class PPOAC2:
    def __call__(self, actor, critic, exp_buffer, config, device='cpu'):
        transform, action_transform = config.data.transforms()

        algo = config.algo
        critic = critic.to(device).train()
        actor = actor.to(device)
        pin_memory = device == 'cuda'
        optim = algo.optimizer.construct(critic.parameters())
        dataset = SARSDataset(exp_buffer, state_transform=transform, action_transform=action_transform)
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True, pin_memory=pin_memory)
        c = Converged(algo.min_change, detections=algo.detections,
                      detection_window=algo.detection_window)

        loss = torch.tensor([-1.0])
        iter = 0

        for state, action, reward, next_state, done in loader:

            state = state.to(device)
            reward = reward.to(device)
            next_state = next_state.to(device)
            done = done.to(device)

            prev_values = np.ones(state.size(0)) * 1000.0
            predicted = np.ones(state.size(0))
            mean_delta_value = np.mean(np.abs(predicted - prev_values))

            #while mean_delta_value > algo.min_change and iter < 5000:
            for i in range(10):

                zero_if_terminal = (~done).to(next_state.dtype)
                next_value = critic(next_state)
                target = reward + zero_if_terminal * config.algo.discount_factor * next_value

                optim.zero_grad()
                predicted = critic(state)
                error = target - predicted
                loss = torch.mean(error ** 2)
                loss.backward()
                optim.step()

                iter += 1
                loss_item = loss.item()
                predicted = predicted.detach().cpu().numpy()
                delta_value = np.abs(predicted - prev_values)
                mean_delta_value = delta_value.max()
                prev_values = predicted

                if iter % algo.logging_freq == 0:
                    logger.info(f'iter     : {iter}')
                    logger.info(f'loss     : {loss_item}')
                    logger.info(f'mean_pred: {delta_value.mean()}')
                    logger.info(f'max_pred : {delta_value.max()}')
                    logger.info(f'min_pred : {delta_value.min()}')


            logger.info(f'iterations: {iter}')
        optim = algo.optimizer.construct(actor.parameters())
        batches_p = 0
        for state, action, reward, next_state, done in loader:
            batches_p += 1

            state = state.to(device)
            reward = reward.to(device)
            next_state = next_state.to(device)
            action = action.to(device)
            done = done.to(device)

            #simple form of the generalized advantage estimate

            with torch.no_grad():
                value = critic(state).squeeze()
                zero_if_terminal = (~done).to(next_state.dtype)
                value_next = zero_if_terminal * critic(next_state).squeeze()
                advantage = reward + value_next - value
                advantage = advantage - advantage.mean() / advantage.std()

            for step in range(config.algo.ppo_steps_per_batch):

                optim.zero_grad()

                new_dist = actor(state)
                new_logprob = new_dist.log_prob(action)
                new_logprob.retain_grad()

                old_dist = actor(state, old=True)
                old_logprob = old_dist.log_prob(action)
                actor.backup()

                loss = ppo_loss_log(new_logprob, old_logprob, advantage, clip=0.2)
                loss.backward()

                optim.step()

        # return a random policy
        return actor, critic



# todo need to get rid of this somehow, just format the observation correctly, or use the transform from the config
def format_observation(observation, features):
    return observation.squeeze().view(-1, features)


class InprobableActionException(Exception):
    pass


def train_ppo_continuous(policy, dataset, config, device='cpu'):

    optim = config.optimizer(config, policy)
    policy = policy.train()
    policy = policy.to(device)

    # batches = math.floor(len(dataset) / config.max_minibatch_size) + 1
    # batch_size = math.floor(len(dataset) / batches)
    # steps_per_batch = math.floor(12 / batches) if math.floor(12 / batches) > 0 else 1

    rollout_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    batches_p = 0
    for i, (observation, action, reward, advantage) in enumerate(rollout_loader):
        batches_p += 1
        for step in range(config.ppo_steps_per_batch):

            observation = observation.to(device)
            advantage = advantage.float().to(device)
            action = action.squeeze().to(device)
            optim.zero_grad()

            # if config.debug:
            #     print(f'ACT__ {action[0].data}')

            mu, sigma = policy(format_observation(observation, policy.features))
            new_dist = torch.distributions.Normal(mu, sigma)
            new_prob = new_dist.log_prob(action)


            #new_prob = torch.distributions.Normal(mu, sigma).log_prob(action).clamp(max=0.0)

            new_prob.retain_grad()
            # if torch.isnan(new_prob).any():
            #     logger.error(new_prob)
            #     logger.error(mu)
            #     logger.error(sigma)
            #     raise InprobableActionException
            old_mu, old_sigma = policy(format_observation(observation, policy.features), old=True)
            old_prob = torch.distributions.Normal(old_mu, old_sigma).log_prob(action)
            #old_prob = torch.distributions.Normal(old_mu, old_sigma).log_prob(action).clamp(max=0.0)
            policy.backup()
            entropy = new_dist.entropy().mean()
            logger.info(f'ENTROPY {entropy.item()}')
            ppo_loss = ppo_loss_log(new_prob, old_prob, advantage, clip=0.2)
            loss = ppo_loss
            loss.backward()
            logger.info(f'LOSS {loss.item()}')
            optim.step()

            # if config.debug:
            #     updated_logprob = policy(observation.squeeze().view(-1, policy.features)).squeeze()
            #     print(f'CHNGE {(torch.exp(updated_logprob) - torch.exp(new_logprob)).data[0]}')
            #     print(f'NEW_G {torch.exp(new_logprob.grad.data[0])}')

    #logging.info(f'processed {batches_p} batches')
    if config.gpu_profile:
        gpu_profile(frame=sys._getframe(), event='line', arg=None)
