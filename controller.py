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
    def __init__(self, r, server_uuid, id, policy, env_config):
        super().__init__()
        self.env_config = env_config
        self._stop_event = threading.Event()
        self.r = r
        self.server_uuid = server_uuid

    def run(self):
        for episode in range(20):
            logging.info(f'starting episode {episode} of {self.env_config.gym_env_string}')
            time.sleep(1)
            EpisodeMessage(self.server_uuid, episode, 1000).send(self.r)
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
            self.rollout_thread = RolloutThread(self.r, self.id, msg.id, msg.policy, msg.env_config)
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


class Trainer(Server):
    def __init__(self, env_config):
        super().__init__()
        self.handler.register(EpisodeMessage, self.episode)
        self.steps = 0
        self.env_config = env_config
        self.config = util.Init(env_config.gym_env_string)
        duallog.setup('logs', 'trainer')

    def episode(self, msg):
        if not self.stopped:
            self.steps += msg.steps
            TrainingProgress(self.id, self.steps).send(self.r)
            logging.info(f'got {self.steps} steps')
            if self.steps > 10000:
                logging.info('got data... sending stop')
                StopMessage(self.id).send(self.r)
                time.sleep(5)
                policy_net = PPOWrap(self.env_config.features, self.env_config.action_map, self.env_config.hidden)
                logging.info('training finished')
                self.steps = 0
                RolloutMessage(self.id, 0, policy_net, self.env_config).send(self.r)

    def stopAll(self, msg):
        super().stopAll(msg)
        self.steps = 0


class GatherThread(threading.Thread):
    def run(self):
        s = Gatherer()
        s.main()


class TrainerThread(threading.Thread):
    def run(self):
        s = Trainer(env_config)
        s.main()


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Start server.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-t", "--trainer", help="start a training instance",
                       action="store_true")
    group.add_argument("-g", "--gatherer", help="start a gathering instance",
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
        trainer = Trainer(env_config)
        trainer.main()
    elif args.gatherer:
        getherer = Gatherer()
        getherer.main()
    elif args.start:
        ResetMessage().send(r)
        RolloutMessage(0, policy_net, env_config).send(r)
    elif args.stopall:
        StopAllMessage().send(r)






    #
    # t1 = TrainerThread()
    # g1 = GatherThread()
    #
    # t1.start()
    # g1.start()
    #
    # r = redis.Redis()
    #
    # RolloutMessage(0, policy_net, env_config).send(r)
    #
    # time.sleep(20)
    #
    # StopAllMessage().send(r)
    #
    # time.sleep(3)
