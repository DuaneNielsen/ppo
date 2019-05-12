import policy_db
import torch
import configs
from models import PPOWrap
import random
import pytest


@pytest.fixture(scope="module")
def db():
    db = policy_db.PolicyDB()
    yield db
    db.delete('run_id')
    db.delete('run1')
    db.delete('run2')
    db.delete('best_run')
    db.delete('delete_me')


def testPostgresWrite(db):

    config = configs.LunarLander()
    s0 = torch.randn(config.features)
    s1 = torch.randn(config.features)
    s = config.prepro(s0, s1).unsqueeze(0)

    policy_net = PPOWrap(config.features, config.action_map, config.hidden)

    a0 = policy_net(s)

    policy_state_dict = policy_net.state_dict()
    policy_net.load_state_dict(policy_state_dict)

    # co = Coordinator(redis_host='localhost', redis_port=6379, redis_db=0, redis_password=None,
    #                  db_host='localhost', db_port=5432, db_name='testpython', db_user='ppo', db_password='password')

    run = "run_id"
    run_state = 'RUNNING'
    stats = {'header': 'unittest', 'ave_reward_episode': 15.3 + random.random()}
    db.write_policy(run, run_state, policy_state_dict, stats, config)
    record = db.get_latest(run)
    policy_net.load_state_dict(record.policy)

    a1 = policy_net(s)

    assert record.run == run
    assert record.stats == stats
    assert torch.equal(a0, a1)
    assert record.config['gym_env_string'] == config.gym_env_string
    assert record.config_b.gym_env_string == config.gym_env_string


def write_policy(db, run, reward):
    config = configs.LunarLander()
    policy_net = PPOWrap(config.features, config.action_map, config.hidden)
    policy_state_dict = policy_net.state_dict()
    run_state = 'RUNNING'
    stats = {'header': 'unittest', 'ave_reward_episode': reward}
    db.write_policy(run, run_state, policy_state_dict, stats, config)


def testDelete(db):
    write_policy(db, 'delete_me', 200.0)

    assert db.count('delete_me') > 0

    db.delete('delete_me')

    assert db.count('delete_me') == 0


def testLatest(db):
    write_policy(db, 'run1', 10.0 + random.random())
    write_policy(db, 'run2', 10.0 + random.random())
    record = db.get_latest()

    assert record.run == 'run2'

    record = db.get_latest('run1')

    assert record.run == 'run1'


def testGetBest(db):
    write_policy(db, 'best_run', 100000.0)
    record = db.get_best('LunarLander-v2')

    assert record.stats['ave_reward_episode'] == 100000.0