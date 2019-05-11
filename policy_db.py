from playhouse.postgres_ext import *
from datetime import datetime
import pickle


database_proxy = Proxy()


class BaseModel(Model):
    """A base model that will use our Postgresql database"""
    class Meta:
        database = database_proxy


class PolicyStore(BaseModel):
    run = CharField()
    timestamp = TimestampField()
    stats = JSONField()
    policy = BlobField()
    config = BlobField()


class PolicyDB:
    def __init__(self, db_host='localhost', db_password='password', db_user='policy_user',  db_name='policy_db', db_port=5432, ):
        db = PostgresqlDatabase(db_name, user=db_user, password=db_password,
                                host=db_host, port=db_port)
        database_proxy.initialize(db)
        db.create_tables([PolicyStore])

    def write_policy(self, policy, config, run, stats):
        row = PolicyStore()
        row.run = "run_id"
        row.stats = "{\"say\":\"hello\"}"
        row.config = pickle.dumps(config, 0)
        row.timestamp = datetime.now()
        policy_weights = policy.state_dict()
        row.policy = pickle.dumps(policy_weights, 0)
        return row.save()

    def get_latest(self, run):
        return PolicyStore.select().where(PolicyStore.run == run).order_by(PolicyStore.timestamp).get()