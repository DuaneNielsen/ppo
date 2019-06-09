from playhouse.postgres_ext import *
from datetime import datetime
import pickle
import json
import copy
import random
import torch

import logging
logger = logging.getLogger(__name__)


policy_db_proxy = Proxy()


class PickleField(BlobField):
    field_type = 'bytea'

    def db_value(self, value):
        return pickle.dumps(value, 0)

    def python_value(self, value):
        return pickle.loads(value)


class PolicyBaseModel(Model):
    """A base model that will use our Postgresql database"""

    class Meta:
        database = policy_db_proxy


class PolicyStore(PolicyBaseModel):
    run = CharField()
    run_state = CharField()
    timestamp = TimestampField(resolution=10 ** 6)
    env_string = CharField()
    iteration = IntegerField()
    reservoir = BooleanField(default=False)
    best = BooleanField(default=False)
    stats = JSONField()
    policy = PickleField()
    config_b = PickleField()


class PolicyDB:
    def __init__(self, db_host='localhost', db_password='password', db_user='policy_user', db_name='policy_db',
                 db_port=5432, ):
        db = PostgresqlDatabase(db_name, user=db_user, password=db_password,
                                host=db_host, port=db_port)
        policy_db_proxy.initialize(db)
        db.create_tables([PolicyStore])

    def calc_next_iteration(self, run):
        record = self.get_latest(run)
        if record is not None:
            return record.iteration + 1
        else:
            return 1

    def write_policy(self, run, run_state, policy_state_dict, stats, config):
        row = PolicyStore()
        row.run = run
        row.env_string = config.gym_env_string
        row.run_state = run_state
        row.iteration = self.calc_next_iteration(run)
        row.stats = stats
        row.config_b = config
        row.timestamp = datetime.now()
        row.policy = policy_state_dict
        return row.save()

    def get_latest(self, run=None):
        try:
            if run is not None:
                return PolicyStore.select().where(PolicyStore.run == run).order_by(-PolicyStore.timestamp).get()
            else:
                return PolicyStore.select().order_by(-PolicyStore.timestamp).get()
        except PolicyStore.DoesNotExist:
            return None

    def get_best(self, env_string=None, run=None):
        if env_string is not None and run is not None:
            return PolicyStore.select().where((PolicyStore.env_string == env_string) &
                                              (PolicyStore.run == run)). \
                order_by(PolicyStore.stats['ave_reward_episode'])

        if env_string is not None and run is None:
            return PolicyStore.select().where(PolicyStore.env_string == env_string). \
                order_by(PolicyStore.stats['ave_reward_episode'])

        if env_string is None and run is not None:
            return PolicyStore.select().where(PolicyStore.run == run). \
                order_by(PolicyStore.stats['ave_reward_episode'])

        return PolicyStore.select().order_by(PolicyStore.stats['ave_reward_episode'])

    def delete(self, run):
        query = PolicyStore.delete().where(PolicyStore.run == run)
        query.execute()

    def count(self, run=None):
        if run is not None:
            return PolicyStore.select().where(PolicyStore.run == run).count()
        return PolicyStore.select().count()

    def reservoir(self, run):
        return PolicyStore.select().where((PolicyStore.run == run) & (PolicyStore.reservoir == True))

    def best(self, run):
        return PolicyStore.select().where((PolicyStore.run == run) & (PolicyStore.best == True)).order_by(-PolicyStore.stats['ave_reward_episode'].cast('float'))

    """
    Call after writing latest record to sample it into the reservoir
    see https://en.wikipedia.org/wiki/Reservoir_sampling
    """

    def update_reservoir(self, run, k=10):
        reservoir = self.reservoir(run)
        depth = reservoir.count()
        latest = self.get_latest(run)
        if depth >= k:
            p = k / latest.iteration
            """  If probability hits, then mark an old record for deletion and mark the new one for retention"""
            if p < random.random():
                old_records = [record for record in reservoir]
                delete_index = random.randrange(len(old_records))
                old_records[delete_index].reservoir = False
                old_records[delete_index].save()
                latest.reservoir = True
                latest.save()
        else:
            latest.reservoir = True
            latest.save()

    def update_best(self, run, n=10):
        PolicyStore.update(best=False).where((PolicyStore.run == run) & (PolicyStore.best == True)).execute()
        top_n = PolicyStore.select().where((PolicyStore.run == run)).order_by(-PolicyStore.stats['ave_reward_episode'].cast('float')).limit(n)
        for record in top_n:
            record.best = True
            record.save()

    """
    Call after updating flags to delete old records
    """

    def prune(self, run):
        latest = self.get_latest(run)
        PolicyStore.delete().where((PolicyStore.best == False) & (PolicyStore.reservoir == False) & (
                PolicyStore.id != latest.id)).execute()

    def runs(self):
        return [record.run for record in PolicyStore.select(PolicyStore.run).distinct()]

    def runs_for_env(self, env_string):
        return [record.run for record in PolicyStore.select(PolicyStore.run).where(PolicyStore.env_string == env_string).distinct()]

    def latest_run(self):
        try:
            record = PolicyStore.select(PolicyStore).order_by(-PolicyStore.timestamp).get()
        except PolicyStore.DoesNotExist:
            return None
        return record

    def set_state_latest(self, state):
        latest = self.latest_run()
        latest.run_state = state
        latest.save()