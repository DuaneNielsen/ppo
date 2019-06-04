import logging
import duallog
from services.server import Server
from data import Db
from messages import StartMessage, StopMessage, EpisodeMessage, TrainCompleteMessage, ConfigUpdateMessage, \
    RolloutMessage, ResetMessage, TrainMessage
from models import *
from policy_db import PolicyDB, PolicyStore
import time

GATHERING = 'GATHERING'
STOPPED = 'STOPPED'
TRAINING = 'TRAINING'

class ModelNotFoundException(Exception):
    pass

class Coordinator(Server):
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0, redis_password=None,
                 db_host='localhost', db_port=5432, db_name='policy_db', db_user='policy_user', db_password=None):
        super().__init__(redis_host, redis_port, redis_db, redis_password)
        self.db = PolicyDB(db_host=db_host, db_port=db_port, db_name=db_name, db_user=db_user, db_password=db_password)
        self.exp_buffer = Db(host=redis_host, port=redis_port, db=redis_db, password=redis_password)

        self.db_host = db_host
        self.db_port = db_port
        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password

        self.handler.register(StartMessage, self.handle_start)
        self.handler.register(StopMessage, self.handle_stop)
        self.handler.register(EpisodeMessage, self.handle_episode)
        self.handler.register(TrainCompleteMessage, self.handle_train_complete)
        self.handler.register(ConfigUpdateMessage, self.handle_config_update)
        self.config = None
        self.policy = None
        self.state = STOPPED
        duallog.setup('logs', 'co-ordinator')
        self.resume(self.db)
        self.last_active = time.time()
        self.start_heartbeat(5, self.heartbeat)

    def resume(self, db):
        record = db.latest_run()
        self.state = record.run_state
        self.config = record.config_b

        self.exp_buffer.clear_rollouts()
        rollout = self.exp_buffer.create_rollout(self.config)
        self.policy = self.init_policy(self.config)
        self.policy.load_state_dict(record.policy)

        if self.state != STOPPED:
            self.state = GATHERING
            RolloutMessage(self.id, rollout.id, self.policy, self.config, self.config.episodes_per_gatherer).send(
                self.r)

    def init_policy(self, config):
        if hasattr(config, 'continuous') and config.continuous:
            if config.model.name == 'MultiPolicyNet':
                model = MultiPolicyNetContinuous(config.model.features_size, config.model.action_size,
                                                 config.model.hidden_size)
            elif config.model.name == 'MultiPolicyNetContinuousV2':
                model = MultiPolicyNetContinuousV2(config.model.features_size, config.model.action_size,
                                                   config.model.hidden_size)
            else:
                raise ModelNotFoundException

            policy = PPOWrapModel(model)
        else:
            policy = PPOWrap(config.features, config.action_map, config.hidden)

        return policy

    def handle_start(self, msg):
        self.state = GATHERING
        self.config = msg.config

        self.policy = self.init_policy(self.config)

        ResetMessage(self.id).send(self.r)

        self.exp_buffer.clear_rollouts()
        rollout = self.exp_buffer.create_rollout(self.config)

        RolloutMessage(self.id, rollout.id, self.policy, self.config, self.config.episodes_per_gatherer).send(self.r)

    def handle_episode(self, msg):
        if not self.state == STOPPED:
            rollout = self.exp_buffer.latest_rollout(self.config)
            steps = len(rollout)
            if steps >= self.config.num_steps_per_rollout and not self.state == TRAINING:
                logging.debug(f'Starting training with {steps} steps')
                rollout.finalize()
                self.state = TRAINING
                total_reward = 0
                for episode in rollout:
                    total_reward += episode.total_reward()
                ave_reward = total_reward / rollout.num_episodes()
                stats = {'ave_reward_episode': ave_reward}
                self.db.write_policy(self.config.run_id, self.state, self.policy.state_dict(), stats, self.config)
                self.db.update_reservoir(self.config.run_id, self.config.policy_reservoir_depth)
                self.db.update_best(self.config.run_id, self.config.policy_top_depth)
                self.db.prune(self.config.run_id)
                TrainMessage(self.id, self.policy, self.config).send(self.r)
        self.last_active = time.time()

    def handle_train_complete(self, msg):
        logging.debug('Training completed')

        self.policy = msg.policy
        ResetMessage(self.id).send(self.r)

        self.exp_buffer.clear_rollouts()
        rollout = self.exp_buffer.create_rollout(self.config)

        if not self.state == STOPPED:
            RolloutMessage(self.id, rollout.id, msg.policy, self.config, self.config.episodes_per_gatherer).send(self.r)
            self.state = GATHERING
        self.last_active = time.time()

    def handle_stop(self, msg):
        logging.debug('Got STOP message')
        self.state = STOPPED
        self.db.set_state_latest(STOPPED)

    def handle_config_update(self, msg):
        logging.debug(f'Got config update')

        run = self.db.latest_run()
        record = self.db.best(run.run).get()
        self.config = msg.config

        self.exp_buffer.clear_rollouts()
        rollout = self.exp_buffer.create_rollout(self.config)

        self.policy = self.init_policy(self.config)
        self.policy.load_state_dict(record.policy)

        self.state = GATHERING

        RolloutMessage(self.id, rollout.id, self.policy, self.config, self.config.episodes_per_gatherer).send(self.r)

    def heartbeat(self):
        time_inactive = time.time() - self.last_active
        logging.debug(
            f'Heartbeat state : {self.state},  time_inactive: {time_inactive}, timeout: {self.config.timeout}')
        if self.state == GATHERING or self.state == TRAINING:
            if time_inactive > self.config.timeout:
                # database connection must be formed inside the thread
                db = PolicyDB(db_host=self.db_host, db_port=self.db_port, db_name=self.db_name,
                              db_user=self.db_user, db_password=self.db_password)
                logging.debug(f'Heartbeat: inactive for {time_inactive} => resuming')
                self.resume(db)
