import redis
import configs
import time
import threading
import util

from messages import StopMessage, StopAllMessage, ResetMessage, StoppedMessage, \
    EpisodeMessage, RolloutMessage, MessageHandler, TrainingProgress
from models import PPOWrap
import uuid
import duallog
import logging
from ppo_clip_discrete import train_policy
from data import RolloutDatasetBase, Db
from rollout import single_episode
import gym
import gym_duane
from tensorboardX import SummaryWriter


class Server:
    def __init__(self, host='localhost', port=6379, db=0):
        self.id = uuid.uuid4()
        self.r = redis.Redis(host, port, db)
        self.handler = MessageHandler(self.r, 'rollout')
        self.stopped = False
        self.handler.register(ResetMessage, self.reset)
        self.handler.register(StopAllMessage, self.stopAll)

    def main(self):
        self.handler.listen()

    def reset(self, _):
        self.stopped = False

    def stopAll(self, msg):
        self.stopped = True
        StoppedMessage(self.id).send(self.r)


class RolloutThread(threading.Thread):
    def __init__(self, redis, server_uuid, policy, env_config, num_episodes=10000):
        super().__init__()
        self.env_config = env_config
        self._stop_event = threading.Event()
        self.redis = redis
        self.db = Db()
        self.server_uuid = server_uuid
        self.policy = policy.to('cpu').eval()
        self.env = gym.make(env_config.gym_env_string)
        self.num_episodes = num_episodes

    def run(self):

        # todo not sure about this way of getting the rollout, might add to a stale rollout by accident
        rollout = self.db.latest_rollout(self.env_config)

        for episode_number in range(1, self.num_episodes+1):
            logging.info(f'starting episode {episode_number} of {self.env_config.gym_env_string}')
            episode = single_episode(self.env, self.env_config, self.policy, rollout)
            EpisodeMessage(self.server_uuid, episode_number, len(episode), episode.total_reward()).send(self.redis)

            if self.stopped():
                logging.info('thread stopped, exiting')
                break

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


class Gatherer(Server):
    def __init__(self):
        super().__init__()
        self.handler.register(RolloutMessage, self.rollout)
        self.handler.register(StopMessage, self.stop)
        self.rollout_thread = None
        duallog.setup('logs', f'gatherer-{self.id}-')

    def rollout(self, msg):
        if not self.stopped:
            self.rollout_thread = RolloutThread(self.r, self.id, msg.policy, msg.env_config)
            self.rollout_thread.start()

    def _stop(self):
        if self.rollout_thread is not None:
            self.rollout_thread.stop()
            self.rollout_thread.join()

    def stop(self, msg):
        self._stop()

    def stopAll(self, msg):
        self._stop()
        super().stopAll(msg)


class DemoThread(threading.Thread):
    def __init__(self, policy, env_config, env, num_episodes=1):
        super().__init__()
        self.env_config = env_config
        self._stop_event = threading.Event()
        self.policy = policy.to('cpu').eval()
        self.num_episodes = num_episodes
        self.env = env

    def run(self):

        for episode_number in range(self.num_episodes):
            logging.info(f'starting episode {episode_number} of {self.env_config.gym_env_string}')
            single_episode(self.env, self.env_config, self.policy, render=True)

            if self.stopped():
                logging.info('thread stopped, exiting')
                break

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


class DemoListener(Server):
    def __init__(self):
        super().__init__()
        self.handler.register(RolloutMessage, self.rollout)
        self.latest_policy = None
        self.demo_thread = None
        self.env = None
        duallog.setup('logs', f'gatherer-{self.id}-')

    def rollout(self, msg):
        if self.env:
            self.env.reset()
        else:
            self.env = gym.make(env_config.gym_env_string)
        self.latest_policy = msg.policy
        if self.demo_thread is None or not self.demo_thread.isAlive():
            self.demo_thread = DemoThread(msg.policy, msg.env_config, self.env)
            self.demo_thread.start()


class Trainer(Server):
    def __init__(self):
        super().__init__()
        self.handler.register(EpisodeMessage, self.episode)
        self.steps = 0
        self.env_config = configs.LunarLander()
        self.config = configs.Init()
        duallog.setup('logs', 'trainer')
        self.db = Db()
        self.policy = PPOWrap(env_config.features, env_config.action_map, env_config.hidden)
        self.rollout = self.db.create_rollout(env_config)

    def episode(self, msg):
        if not self.stopped:
            self.steps += msg.steps
            TrainingProgress(self.id, self.steps).send(self.r)
            logging.info(f'got {self.steps} steps')
            if self.steps > 10000:
                self.rollout.end()
                dataset = RolloutDatasetBase(env_config, self.rollout)

                logging.info('got data... sending stop')
                StopMessage(self.id).send(self.r)

                logging.info('started training')
                train_policy(self.policy, dataset, self.config)
                logging.info('training finished')

                logging.info('deleting rollout')
                self.db.delete_rollout(self.rollout)
                self.steps = 0
                self.rollout = self.db.create_rollout(env_config)
                logging.info('deleted rollout')

                logging.info('starting next rollout')
                RolloutMessage(self.id, self.rollout.id, self.policy, self.env_config).send(self.r)

    def stopAll(self, msg):
        super().stopAll(msg)
        self.steps = 0


class TensorBoardListener(Server):
    def __init__(self):
        super().__init__()
        self.env_config = configs.LunarLander()
        self.handler.register(EpisodeMessage, self.episode)
        self.handler.register(StopAllMessage, self.stopAll)
        self.tb_step = 0
        self.tb = self.config.getSummaryWriter(env_config.gym_env_string)

    def episode(self, msg):
        self.tb.add_scalar('reward', msg.total_reward, self.tb_step)
        self.tb.add_scalar('epi_len', msg.steps, self.tb_step)
        self.tb_step += 1

    def stopAll(self, msg):
        self.tb_step = 0
        self.tb = self.config.getSummaryWriter(env_config.gym_env_string)




class GatherThread(threading.Thread):
    def run(self):
        s = Gatherer()
        s.main()


class TrainerThread(threading.Thread):
    def run(self):
        s = Trainer()
        s.main()


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Start server.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-t", "--trainer", help="start a training instance",
                       action="store_true")
    group.add_argument("-g", "--gatherer", help="start a gathering instance",
                       action="store_true")
    group.add_argument("-m", "--monitor", help="start a monitoring instance",
                       action="store_true")
    group.add_argument("-d", "--demo", help="start a demo instance",
                       action="store_true")
    group.add_argument("--start", help="start training",
                       action="store_true")
    group.add_argument("--stopall", help="stop training",
                       action="store_true")
    args = parser.parse_args()

    env_config = configs.LunarLander()
    policy_net = PPOWrap(env_config.features, env_config.action_map, env_config.hidden)

    r = redis.Redis()

    if args.trainer:
        trainer = Trainer()
        trainer.main()
    elif args.gatherer:
        gatherer = Gatherer()
        gatherer.main()
    elif args.monitor:
        tb = TensorBoardListener()
        tb.main()

    elif args.demo:
        demo = DemoListener()
        demo.main()

    elif args.start:
        ResetMessage(uuid.uuid4()).send(r)
        RolloutMessage(0, policy_net, env_config).send(r)
    elif args.stopall:
        StopAllMessage(uuid.uuid4()).send(r)