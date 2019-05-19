import redis
import configs
import time
import threading

from messages import *
from models import PPOWrap
import duallog
import logging
from ppo_clip_discrete import train_policy
from data import RolloutDatasetBase, Db
from rollout import single_episode
import gym
import uuid
from policy_db import PolicyDB
import concurrent.futures


class Server:
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0, redis_password=None, redis_client=None):
        self.id = uuid.uuid4()

        if redis_client is not None:
            self.r = redis_client
        else:
            self.r = redis.Redis(host=redis_host, port=redis_port, db=redis_db, password=redis_password)

        self.handler = MessageHandler(self.r, 'rollout')
        self.handler.register(ExitMessage, self.exit)

    def main(self):
        self.handler.listen()

    def exit(self, msg):
        raise SystemExit


class Gatherer(Server):
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0, redis_password=None):
        super().__init__(redis_host, redis_port, redis_db, redis_password)
        self.handler.register(RolloutMessage, self.rollout)
        self.rollout_thread = None
        self.exp_buffer = Db(host=redis_host, port=redis_port, password=redis_password)
        self.redis = self.exp_buffer.redis
        self.job = None

        duallog.setup('logs', f'gatherer-{self.id}-')

    def rollout(self, msg):

        policy = msg.policy.to('cpu').eval()
        env = gym.make(msg.config.gym_env_string)
        rollout = self.exp_buffer.rollout(msg.rollout_id, msg.config)
        episode_number = 0

        while len(rollout) < msg.config.num_steps_per_rollout:
            logging.info(f'starting episode {episode_number} of {msg.config.gym_env_string}')
            episode = single_episode(env, msg.config, policy, rollout)
            EpisodeMessage(self.id, episode_number, len(episode), episode.total_reward()).send(self.redis)
            episode_number += 1


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


class DemoListener(Server):
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0, redis_password=None):
        super().__init__(redis_host, redis_port, redis_db, redis_password)
        #todo need a new message for demo server here
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
        self.handler.register(TrainMessage, self.handle_train)
        duallog.setup('logs', 'trainer')
        self.exp_buffer = Db(host=redis_host, port=redis_port, db=redis_db, password=redis_password)

    def handle_train(self, msg):
        rollout = self.exp_buffer.latest_rollout(msg.config)
        assert len(rollout) != 0
        dataset = RolloutDatasetBase(config, rollout)
        policy = msg.policy

        logging.info('started training')
        train_policy(policy, dataset, msg.config)
        logging.info('training finished')

        logging.info('training complete')
        TrainCompleteMessage(self.id, policy, msg.config).send(self.r)


GATHERING = 'GATHERING'
STOPPED = 'STOPPED'
TRAINING = 'TRAINING'


class Coordinator(Server):
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0, redis_password=None,
                 db_host='localhost', db_port=5432, db_name='policy_db', db_user='policy_user', db_password=None):
        super().__init__(redis_host, redis_port, redis_db, redis_password)
        self.db = PolicyDB(db_host=db_host, db_port=db_port, db_name=db_name, db_user=db_user, db_password=db_password)
        self.exp_buffer = Db(host=redis_host, port=redis_port, db=redis_db, password=redis_password)

        self.handler.register(StartMessage, self.handle_start)
        self.handler.register(StopMessage, self.handle_stop)
        self.handler.register(EpisodeMessage, self.handle_episode)
        self.handler.register(TrainCompleteMessage, self.handle_train_complete)
        self.run_id = None
        self.config = None
        self.policy = None
        self.state = STOPPED
        duallog.setup('logs', 'co-ordinator')

    def handle_start(self, msg):
        self.state = GATHERING
        self.config = msg.config
        self.policy = PPOWrap(self.config.features, self.config.action_map, self.config.hidden)
        self.run_id = config.rundir(config.gym_env_string)

        ResetMessage(self.id).send(r)

        rollout = self.exp_buffer.latest_rollout(self.config)
        if rollout is not None:
            self.exp_buffer.delete_rollout(rollout)
        rollout = self.exp_buffer.create_rollout(self.config)

        RolloutMessage(self.id, rollout.id, self.policy, self.config, self.config.episodes_per_gatherer).send(r)

    def handle_episode(self, msg):
        if not self.state == STOPPED:
            rollout = self.exp_buffer.latest_rollout(self.config)
            steps = len(rollout)
            logging.debug(int(steps))
            if steps > config.num_steps_per_rollout and not self.state == TRAINING:
                rollout.finalize()
                self.state = TRAINING
                total_reward = 0
                for episode in rollout:
                    total_reward += episode.total_reward()
                ave_reward = total_reward / rollout.num_episodes()
                stats = {'ave_reward_per_episode': ave_reward}
                self.db.write_policy(self.run_id, self.state, self.policy.state_dict(), stats, config)
                self.db.update_reservoir(self.run_id, config.policy_reservoir_depth)
                self.db.update_best(self.run_id, config.policy_top_depth)
                self.db.prune(self.run_id)
                TrainMessage(self.id, self.policy, self.config).send(self.r)

    def handle_train_complete(self, msg):
        self.policy = msg.policy
        ResetMessage(self.id).send(r)

        rollout = self.exp_buffer.latest_rollout(self.config)
        self.exp_buffer.delete_rollout(rollout)
        rollout = self.exp_buffer.create_rollout(self.config)

        if not self.state == STOPPED:
            RolloutMessage(self.id, rollout.id, msg.policy, self.config, self.config.episodes_per_gatherer).send(r)
            self.state = GATHERING

    def handle_stop(self, msg):
        self.state = STOPPED


class TensorBoardListener(Server):
    def __init__(self, redis_host, redis_port, redis_db, redis_password):
        super().__init__(redis_host, redis_port, redis_db, redis_password)
        # todo fix the hardcoded config
        self.config = configs.LunarLander()
        self.handler.register(EpisodeMessage, self.episode)
        self.tb_step = 0
        # todo probably needs to be moved again!
        self.tb = self.config.getSummaryWriter(config.gym_env_string)

    def episode(self, msg):
        self.tb.add_scalar('reward', msg.total_reward, self.tb_step)
        self.tb.add_scalar('epi_len', msg.steps, self.tb_step)
        self.tb_step += 1


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

    parser.add_argument("-ph", "--postgres-host", help='hostname of postgres server', dest='postgres_host',
                        default='localhost')
    parser.add_argument("-pp", "--postgres-port", help='port of postgres server', dest='postgres_port', default=5432)
    parser.add_argument("-pd", "--postgres-db", help='hostname of postgres db', dest='postgres_db',
                        default='policy_db')
    parser.add_argument("-pu", "--postgres-user", help='hostname of postgres user', dest='postgres_user',
                        default='policy_user')
    parser.add_argument("-pa", "--postgres-password", help='password of postgres server', dest='postgres_password',
                        default='password')

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
        trainer = Trainer(args.redis_host, args.redis_port, args.redis_db, args.redis_password)
        trainer.main()

    elif args.gatherer:
        gatherer = Gatherer(args.redis_host, args.redis_port, args.redis_db, args.redis_password)
        gatherer.main()

    elif args.monitor:
        tb = TensorBoardListener(args.redis_host, args.redis_port, args.redis_db, args.redis_password)
        tb.main()

    elif args.demo:
        demo = DemoListener(args.redis_host, args.redis_port, args.redis_db, args.redis_password)
        demo.main()

    elif args.coordinator:
        demo = Coordinator(args.redis_host, args.redis_port, args.redis_db, args.redis_password,
                           args.postgres_host, args.postgres_port, args.postgres_db, args.postgres_user,
                           args.postgres_password)
        demo.main()

    elif args.start:
        ResetMessage(uuid.uuid4()).send(r)
        RolloutMessage(uuid.uuid4(), 0, policy_net, config).send(r)
    elif args.stopall:
        StopAllMessage(uuid.uuid4()).send(r)
