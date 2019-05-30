import pytest
from services.server import Server
from services.gatherer import Gatherer
import redis
from messages import *
from threading import Thread
from data import Db
from configs import LunarLander, HalfCheetah
from models import PPOWrap, PPOWrapModel, MultiPolicyNetContinuous
from time import sleep


class ServerThread(Thread):
    def __init__(self, s):
        super().__init__()
        self.s = s

    def run(self):
        self.s.main()


@pytest.fixture
def r():
    return redis.Redis('localhost')


def test_server_exit(r):
    s = Server('localhost', 6379, 0, None)
    ServerThread(s).start()
    ExitMessage(s.id).send(r)


def test_gatherer_exit(r):
    s = Gatherer()
    ServerThread(s).start()
    ExitMessage(s.id).send(r)


class TestServer(Server):
    def __init__(self, r):
        super().__init__(redis_client=r)
        self.handler.register(EpisodeMessage, self.episode)

    def episode(self, msg):
        print(msg.steps)


def test_gatherer_processing(r):
    s = Gatherer()
    ServerThread(s).start()

    tst = TestServer(r)
    ServerThread(tst).start()

    config = LunarLander()
    policy_net = PPOWrap(config.features, config.action_map, config.hidden)
    db = Db(redis_client=r)
    rollout = db.create_rollout(config)

    RolloutMessage(s.id, rollout.id, policy_net, config, 10).send(r)

    sleep(5)

    assert len(rollout) >= config.num_steps_per_rollout
    db.delete_rollout(rollout)
    ExitMessage(s.id).send(r)


def test_gatherer_processing_continuous(r):
    s = Gatherer()
    ServerThread(s).start()

    tst = TestServer(r)
    ServerThread(tst).start()

    config = HalfCheetah()
    model = MultiPolicyNetContinuous(config.model.features_size, config.model.action_size, config.model.hidden_size)
    policy_net = PPOWrapModel(model)
    db = Db(redis_client=r)
    rollout = db.create_rollout(config)

    RolloutMessage(s.id, rollout.id, policy_net, config, 10).send(r)

    sleep(5)

    assert len(rollout) >= config.num_steps_per_rollout
    db.delete_rollout(rollout)
    ExitMessage(s.id).send(r)


def test_gatherer_ping(r):
    s = Gatherer()
    ServerThread(s).start()

    PingMessage(s.id).send(r)

    sleep(1)

    ExitMessage(s.id).send(r)