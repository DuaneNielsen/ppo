import pytest
from services.server import Server
import redis
from messages import *
from threading import Thread
from data import Db
from configs import Discrete, Continuous
from time import sleep
from services import *
from uuid import uuid4
import logging

logging.basicConfig(format='%(levelname)s-%(module)s-%(message)s', level=logging.DEBUG)
logging.getLogger('peewee').setLevel(logging.INFO)
logging.getLogger('coordinator').setLevel(logging.INFO)
logging.getLogger('server').setLevel(logging.DEBUG)
logging.getLogger('trainer').setLevel(logging.DEBUG)
logging.getLogger('gatherer').setLevel(logging.DEBUG)


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


class ListenTestServer(Server):
    def __init__(self, r):
        super().__init__(redis_client=r)
        self.handler.register(EpisodeMessage, self.episode)
        self.handler.register(TrainCompleteMessage, self.handle_train_complete)

    def episode(self, msg):
        print(f'Episode {msg.steps}')

    def handle_train_complete(self, msg):
        print('Training complete')
        ExitMessage(self.id).send(self.r)


def test_gatherer_processing(r):
    config = Discrete('LunarLander-v2')
    db = Db(redis_client=r)
    rollout = db.create_rollout(config.data.coder)
    s = Gatherer()
    tst = ListenTestServer(r)

    try:
        ServerThread(s).start()
        ServerThread(tst).start()

        random_policy = config.random_policy.construct()

        RolloutMessage(s.id, 'LunarLander-v2_xxx', rollout.id, random_policy, config, 1).send(r)
        sleep(5)
        assert len(rollout) >= config.gatherer.num_steps_per_rollout

    finally:
        db.delete_rollout(rollout)
        ExitMessage(s.id).send(r)


def test_gatherer_processing_continuous(r):

    config = Continuous('RoboschoolHalfCheetah-v1')
    db = Db(redis_client=r)
    rollout = db.create_rollout(config.data.coder)
    s = Gatherer()
    tst = ListenTestServer(r)

    try:
        ServerThread(s).start()
        ServerThread(tst).start()

        random_policy = config.random_policy.construct()

        RolloutMessage(s.id, 'HalfCheetah-v2_xxx', rollout.id, random_policy, config, 1).send(r)
        sleep(5)
        assert len(rollout) >= config.gatherer.num_steps_per_rollout

    finally:
        db.delete_rollout(rollout)
        ExitMessage(s.id).send(r)


def test_gatherer_ping(r):
    s = Gatherer()
    ServerThread(s).start()

    PingMessage(s.id).send(r)

    sleep(1)

    ExitMessage(s.id).send(r)


def test_trainer_exit(r):
    ServerThread(Trainer()).start()
    ServerThread(ListenTestServer(r)).start()
    ExitMessage(server_uuid=uuid4()).send(r)


def test_trainer(r):

    config = Discrete('LunarLander-v2')
    db = Db(redis_client=r)
    rollout = db.create_rollout(config.data.coder.construct())
    g = Gatherer()
    t = Trainer()
    tst = ListenTestServer(r)
    uuid = uuid4()

    try:
        ServerThread(g).start()
        ServerThread(t).start()
        ServerThread(tst).start()

        random_policy = config.random_policy.construct()
        actor = config.actor.construct()
        critic = config.critic.construct()

        RolloutMessage(uuid, 'LunarLander-v2_xxx', rollout.id, random_policy, config, 1).send(r)

        sleep(3)
        assert len(rollout) >= config.gatherer.num_steps_per_rollout
        TrainMessage(uuid, actor, critic, config).send(r)

        sleep(50)

    finally:
        db.delete_rollout(rollout)
        ExitMessage(uuid).send(r)