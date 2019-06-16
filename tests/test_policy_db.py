import policy_db
import torch
import configs
from models import PPOWrapModel
import random
import pytest


def delete(db):
    db.delete('run_id')
    db.delete('run1')
    db.delete('run2')
    db.delete('best_run')
    db.delete('delete_me')


@pytest.fixture(scope="module")
def db():
    db = policy_db.PolicyDB()
    yield db
    delete(db)


def testPostgresWrite(db):

    config = configs.default
    s0 = torch.randn(config.env.state_space_shape)
    s1 = torch.randn(config.env.state_space_shape)
    s = config.data.prepro(s0, s1).unsqueeze(0)

    actor = config.actor.construct()
    actor = PPOWrapModel(actor)
    critic = config.critic.construct()

    a0 = actor(s)

    actor_state_dict = actor.state_dict()
    actor.load_state_dict(actor_state_dict)

    critic_state_dict = critic.state_dict()

    # co = Coordinator(redis_host='localhost', redis_port=6379, redis_db=0, redis_password=None,
    #                  db_host='localhost', db_port=5432, db_name='testpython', db_user='ppo', db_password='password')

    run = "run_id"
    run_state = 'RUNNING'
    stats = {'header': 'unittest', 'ave_reward_episode': 15.3 + random.random()}
    db.write_policy(run, run_state, actor_state_dict, critic_state_dict, stats, config)
    record = db.get_latest(run)
    actor.load_state_dict(record.actor)

    a1 = actor(s)

    assert record.run == run
    assert record.stats == stats
    assert a0.entropy() == a1.entropy()
    assert record.config_b.env.name == config.env.name


def write_policy(db, run, reward):
    config = configs.default
    actor = config.actor.construct()
    critic = config.critic.construct()
    actor = PPOWrapModel(actor)
    actor_state_dict = actor.state_dict()
    critic_state_dict = critic.state_dict()
    run_state = 'RUNNING'
    stats = {'header': 'unittest', 'ave_reward_episode': reward}
    db.write_policy(run, run_state, actor_state_dict, critic_state_dict, stats, config)


def test_delete(db):
    write_policy(db, 'delete_me', 200.0)

    assert db.count('delete_me') > 0

    db.delete('delete_me')

    assert db.count('delete_me') == 0


def test_latest(db):
    write_policy(db, 'run1', 10.0 + random.random())
    write_policy(db, 'run2', 10.0 + random.random())
    record = db.get_latest()

    assert record.run == 'run2'

    record = db.get_latest('run1')

    assert record.run == 'run1'


def test_get_best(db):
    write_policy(db, 'best_run', 100000.0)
    record = db.get_best(env_string='LunarLander-v2').get()
    assert record.stats['ave_reward_episode'] == 100000.0
    record = db.get_best(run='best_run').get()
    assert record.stats['ave_reward_episode'] == 100000.0
    record = db.get_best(env_string='LunarLander-v2', run='best_run').get()
    assert record.stats['ave_reward_episode'] == 100000.0
    try:
        record = db.get_best(env_string='LunarLander-v1', run='best_run').get()
        assert False
    except policy_db.PolicyStore.DoesNotExist:
        assert True


def test_update_best(db):

    delete(db)

    write_policy(db, 'best_run', 1.0)
    write_policy(db, 'best_run', 2.0)
    write_policy(db, 'best_run', 3.0)
    write_policy(db, 'best_run', 4.0)

    assert db.best('best_run').count() == 0

    db.update_best('best_run', n=3)
    db.prune('best_run')

    assert db.best('best_run').count() == 3
    assert db.best('best_run').get().stats['ave_reward_episode'] == 4.0

    write_policy(db, 'best_run', 5.0)

    assert db.best('best_run').count() == 3

    db.update_best('best_run', n=3)
    db.prune('best_run')

    assert db.best('best_run').count() == 3
    assert db.best('best_run').get().stats['ave_reward_episode'] == 5.0

def test_iteration(db):

    delete(db)

    assert db.calc_next_iteration('run1') == 1
    assert db.calc_next_iteration('run2') == 1
    write_policy(db, 'run1', 10.0)
    assert db.calc_next_iteration('run1') == 2
    assert db.calc_next_iteration('run2') == 1
    write_policy(db, 'run2', 10.0)
    assert db.calc_next_iteration('run1') == 2
    assert db.calc_next_iteration('run2') == 2


def test_count(db):

    delete(db)

    assert db.count() == 0
    assert db.count('run1') == 0
    assert db.count('run2') == 0

    write_policy(db, 'run1', 10.0)
    assert db.count() == 1
    assert db.count('run1') == 1
    assert db.count('run2') == 0

    write_policy(db, 'run2', 10.0)
    assert db.count() == 2
    assert db.count('run1') == 1
    assert db.count('run2') == 1


def test_update_reservoir(db):

    delete(db)

    assert db.reservoir('run1').count() == 0
    write_policy(db, 'run1', 10.0)
    assert db.reservoir('run1').count() == 0
    db.update_reservoir('run1', 3)
    assert db.reservoir('run1').count() == 1
    assert db.count('run1') == 1
    db.prune('run1')
    assert db.count('run1') == 1
    assert db.reservoir('run1').count() == 1

    write_policy(db, 'run1', 10.0)
    assert db.reservoir('run1').count() == 1
    db.update_reservoir('run1', 3)
    assert db.reservoir('run1').count() == 2
    assert db.count('run1') == 2
    db.prune('run1')
    assert db.count('run1') == 2
    assert db.reservoir('run1').count() == 2


    write_policy(db, 'run1', 10.0)
    assert db.reservoir('run1').count() == 2
    db.update_reservoir('run1', 3)
    assert db.reservoir('run1').count() == 3
    assert db.count('run1') == 3
    db.prune('run1')
    assert db.count('run1') == 3
    assert db.reservoir('run1').count() == 3

    write_policy(db, 'run1', 10.0)
    assert db.reservoir('run1').count() == 3
    db.update_reservoir('run1', 3)
    assert db.reservoir('run1').count() == 3
    assert db.count('run1') == 4
    db.prune('run1')
    assert db.count('run1') == 4 or db.count('run1') == 3
    assert db.reservoir('run1').count() == 3

    write_policy(db, 'run1', 10.0)
    assert db.reservoir('run1').count() == 3
    db.update_reservoir('run1', 3)
    assert db.reservoir('run1').count() == 3
    assert db.count('run1') == 4 or db.count('run1') == 5
    db.prune('run1')
    assert db.count('run1') == 4 or db.count('run1') == 3
    assert db.reservoir('run1').count() == 3

    write_policy(db, 'run1', 10.0)
    assert db.reservoir('run1').count() == 3
    db.update_reservoir('run1', 3)
    assert db.reservoir('run1').count() == 3
    assert db.count('run1') == 4 or db.count('run1') == 5
    db.prune('run1')
    assert db.count('run1') == 4 or db.count('run1') == 3
    assert db.reservoir('run1').count() == 3

    write_policy(db, 'run1', 10.0)
    assert db.reservoir('run1').count() == 3
    db.update_reservoir('run1', 3)
    assert db.reservoir('run1').count() == 3
    assert db.count('run1') == 4 or db.count('run1') == 5
    db.prune('run1')
    assert db.count('run1') == 4 or db.count('run1') == 3
    assert db.reservoir('run1').count() == 3


def test_runs(db):
    write_policy(db, 'run1', 10.0)
    assert len(db.runs()) == 1
    assert db.runs()[0] == 'run1'
    write_policy(db, 'run1', 10.0)
    assert len(db.runs()) == 1
    assert db.runs()[0] == 'run1'
    write_policy(db, 'run2', 10.0)
    assert len(db.runs()) == 2
    runs =db.runs()
    for run in db.runs():
        assert (run == 'run1' or run == 'run2')

