import redis
import configs
import time
import threading

from messages import StopMessage, StopAllMessage, ResetMessage, StoppedMessage, \
    EpisodeMessage, RolloutMessage, MessageHandler, TrainingProgress
from models import PPOWrap
import duallog
import logging
from ppo_clip_discrete import train_policy
from data import RolloutDatasetBase, Db
from rollout import single_episode
import gym
import uuid
from policy_db import PolicyDB


class Server:
    def __init__(self, redis_host, redis_port, redis_db, redis_password):
        self.id = uuid.uuid4()
        self.r = redis.Redis(host=redis_host, port=redis_port, db=redis_db, password=redis_password)
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
    def __init__(self, redis, server_uuid, policy, config, num_episodes=10000):
        super().__init__()
        self.config = config
        self._stop_event = threading.Event()
        self.redis = redis
        logging.debug(f'connecting to redis on host: {config.redis_host} port: {config.redis_port} pass: {config.redis_password}')
        self.db = Db(host=config.redis_host, port=config.redis_port, password=config.redis_password)
        self.server_uuid = server_uuid
        self.policy = policy.to('cpu').eval()
        self.env = gym.make(config.gym_env_string)
        self.num_episodes = num_episodes

    def run(self):

        # todo not sure about this way of getting the rollout, might add to a stale rollout by accident
        rollout = self.db.latest_rollout(self.config)

        for episode_number in range(1, self.num_episodes + 1):
            logging.info(f'starting episode {episode_number} of {self.config.gym_env_string}')
            episode = single_episode(self.env, self.config, self.policy, rollout)
            EpisodeMessage(self.server_uuid, episode_number, len(episode), episode.total_reward()).send(self.redis)

            if self.stopped():
                logging.info('thread stopped, exiting')
                break

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


class Gatherer(Server):
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0, redis_password=None):
        super().__init__(redis_host, redis_port, redis_db, redis_password)
        self.handler.register(RolloutMessage, self.rollout)
        self.handler.register(StopMessage, self.stop)
        self.rollout_thread = None
        duallog.setup('logs', f'gatherer-{self.id}-')

    def rollout(self, msg):
        if not self.stopped:
            self.rollout_thread = RolloutThread(self.r, self.id, msg.policy, msg.config)
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
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0, redis_password=None):
        super().__init__(redis_host, redis_port, redis_db, redis_password)
        self.handler.register(RolloutMessage, self.rollout)
        self.latest_policy = None
        self.demo_thread = None
        self.env = None
        duallog.setup('logs', f'demo_listener-{self.id}-')

    def rollout(self, msg):
        if self.env:
            self.env.reset()
        else:
            self.env = gym.make(config.gym_env_string)
        self.latest_policy = msg.policy
        if self.demo_thread is None or not self.demo_thread.isAlive():
            self.demo_thread = DemoThread(msg.policy, msg.env_config, self.env)
            self.demo_thread.start()


class Trainer(Server):
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0, redis_password=None):
        super().__init__(redis_host, redis_port, redis_db, redis_password)
        self.handler.register(EpisodeMessage, self.episode)
        self.steps = 0
        self.config = config
        duallog.setup('logs', 'trainer')
        self.db = Db(host=redis_host, port=redis_port, db=redis_db, password=redis_password)
        self.policy = PPOWrap(config.features, config.action_map, config.hidden)
        self.rollout = self.db.create_rollout(config)

    def episode(self, msg):
        if not self.stopped:
            self.steps += msg.steps
            TrainingProgress(self.id, self.steps).send(self.r)
            logging.info(f'got {self.steps} steps')
            if self.steps > 10000:
                self.rollout.end()
                dataset = RolloutDatasetBase(config, self.rollout)

                logging.info('got data... sending stop')
                StopMessage(self.id).send(self.r)

                logging.info('started training')
                train_policy(self.policy, dataset, self.config)
                logging.info('training finished')

                logging.info('deleting rollout')
                self.db.delete_rollout(self.rollout)
                self.steps = 0
                self.rollout = self.db.create_rollout(config)
                logging.info('deleted rollout')

                logging.info('starting next rollout')
                RolloutMessage(self.id, self.rollout.id, self.policy, self.config).send(self.r)

    def stopAll(self, msg):
        super().stopAll(msg)
        self.steps = 0


class Coordinator(Server):
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0, redis_password=None,
                 db_host='localhost', db_port=5432, db_name='policy_db', db_user='policy_user', db_password=None):
        super().__init__(redis_host, redis_port, redis_db, redis_password)
        self.db = PolicyDB(db_host=db_host, db_port=db_port, db_name=db_name, db_user=db_user, db_password=db_password)





class TensorBoardListener(Server):
    def __init__(self, redis_host, redis_port, redis_db, redis_password):
        super().__init__(redis_host, redis_port, redis_db, redis_password)
        #todo fix the hardcoded config
        self.env_config = configs.LunarLander()
        self.handler.register(EpisodeMessage, self.episode)
        self.handler.register(StopAllMessage, self.stopAll)
        self.tb_step = 0
        #todo probably needs to be moved again!
        self.tb = self.config.getSummaryWriter(config.gym_env_string)

    def episode(self, msg):
        self.tb.add_scalar('reward', msg.total_reward, self.tb_step)
        self.tb.add_scalar('epi_len', msg.steps, self.tb_step)
        self.tb_step += 1

    def stopAll(self, msg):
        self.tb_step = 0
        #todo need to rethink how I'm doing this
        self.tb = self.config.getSummaryWriter(config.gym_env_string)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Start server.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-t", "--trainer", help="start a training instance",
                       action="store_true")
    group.add_argument("-c", "--coordinator", help="start a coordinator instance",
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
    parser.add_argument("-rh", "--redis-host", help='hostname of redis server', dest='redis_host', default='localhost')
    parser.add_argument("-rp", "--redis-port", help='port of redis server', dest='redis_port', default=6379)
    parser.add_argument("-ra", "--redis-password", help='password of redis server', dest='redis_password', default=None)
    parser.add_argument("-rd", "--redis-db", help='db of redis server', dest='redis_db', default=0)
    parser.add_argument("-ph", "--postgres-host", help='hostname of postgres server', dest='postgres_host', default='localhost')
    parser.add_argument("-pp", "--postgres-port", help='port of postgres server', dest='postgres_port', default=6379)
    parser.add_argument("-pd", "--postgres-db", help='hostname of postgres db', dest='postgres_db',
                        default='policy_store')
    parser.add_argument("-pu", "--postgres-user", help='hostname of postgres user', dest='postgres_user',
                        default='pu')
    parser.add_argument("-pa", "--postgres-password", help='password of postgres server', dest='postgres_password', default=None)

    args = parser.parse_args()

    print(args)

    config = configs.LunarLander()

    policy_net = PPOWrap(config.features, config.action_map, config.hidden)

    while True:
        try:
            r = redis.Redis(host=args.redis_host, port=args.redis_port, password=args.redis_password)
            break
        except:
            logging.ERROR(f'connecting to redis on {args.redis_host}:{args.redis_port} waiting 10 secs and retrying')
            time.sleep(10.0)

    if args.trainer:
        trainer = Trainer(args.redis_host, args.redis_port, args.redis_db, args.redis.redis_password)
        trainer.main()

    elif args.gatherer:
        gatherer = Gatherer(args.redis_host, args.redis_port, args.redis_db, args.redis.redis_password)
        gatherer.main()

    elif args.monitor:
        tb = TensorBoardListener(args.redis_host, args.redis_port, args.redis_db, args.redis.redis_password)
        tb.main()

    elif args.demo:
        demo = DemoListener(args.redis_host, args.redis_port, args.redis_db, args.redis.redis_password)
        demo.main()

    elif args.coordinator:
        demo = Coordinator(args.redis_host, args.redis_port, args.redis_db, args.redis_password,
                           args.postgres_host, args.postgres_port, args.postgres_db, args.postgres_user, args.postgres_password)
        demo.main()

    elif args.start:
        ResetMessage(uuid.uuid4()).send(r)
        RolloutMessage(uuid.uuid4(), 0, policy_net, config).send(r)
    elif args.stopall:
        StopAllMessage(uuid.uuid4()).send(r)
