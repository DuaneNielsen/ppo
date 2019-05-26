import logging

import gym

import duallog
from services.server import Server
from data import Db
from messages import RolloutMessage, EpisodeMessage
from rollout import single_episode


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
            EpisodeMessage(self.id, episode_number, len(episode), episode.total_reward(),
                           msg.config.num_steps_per_rollout).send(self.redis)
            episode_number += 1