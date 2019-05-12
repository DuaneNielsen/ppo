from playhouse.postgres_ext import *
from datetime import datetime
import pickle
import json
import configs

policy_db_proxy = Proxy()


class PickleField(BlobField):
    field_type = 'bytea'

    def db_value(self, value):
        return pickle.dumps(value, 0)

    def python_value(self, value):
        return pickle.loads(value)


class ConfigField(JSONField):
    field_type = 'json'

    def db_value(self, value):

        attribs = vars(value)

        # attributes excluded from JSON
        if 'step_coder' in attribs:
            del attribs['step_coder']
        if 'prepro' in attribs:
            del attribs['prepro']
        if 'transform' in attribs:
            del attribs['transform']

        return json.dumps(attribs)

    def python_value(self, value):
        return value


class PolicyBaseModel(Model):
    """A base model that will use our Postgresql database"""
    class Meta:
        database = policy_db_proxy


class PolicyStore(PolicyBaseModel):
    run = CharField()
    run_state = CharField()
    timestamp = TimestampField()
    stats = JSONField()
    policy = PickleField()
    config_b = PickleField()
    config = ConfigField()


class PolicyDB:
    def __init__(self, db_host='localhost', db_password='password', db_user='policy_user',  db_name='policy_db', db_port=5432, ):
        db = PostgresqlDatabase(db_name, user=db_user, password=db_password,
                                host=db_host, port=db_port)
        policy_db_proxy.initialize(db)
        db.create_tables([PolicyStore])

    def write_policy(self, run, run_state, policy_state_dict, stats, config):
        row = PolicyStore()
        row.run = run
        row.run_state = run_state
        row.stats = stats
        row.config = config
        row.config_b = config
        row.timestamp = datetime.now()
        row.policy = policy_state_dict
        return row.save()

    def get_latest(self, run):
        return PolicyStore.select().where(PolicyStore.run == run).order_by(-PolicyStore.timestamp).get()

    def get_best(self, env_string):
        return PolicyStore.select().where(PolicyStore.config['gym_env_string'] == env_string).\
            order_by(PolicyStore.stats['ave_reward_episode']).get()