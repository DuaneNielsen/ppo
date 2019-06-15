from services import Trainer
from messages import *
from services.server import Server
import redis
import pytest
from uuid import uuid4
from threading import Thread


class ServerThread(Thread):
    def __init__(self, s):
        super().__init__()
        self.s = s

    def run(self):
        self.s.main()


@pytest.fixture
def r():
    return redis.Redis('localhost')


class ListenTestServer(Server):
    def __init__(self, r):
        super().__init__(redis_client=r)
        self.handler.register(TrainCompleteMessage, self.handle_train_complete)

    def handle_train_complete(self, msg):
        print(msg)


def test_trainer_exit(r):
    ServerThread(Trainer()).start()
    ServerThread(ListenTestServer(r)).start()
    ExitMessage(server_uuid=uuid4()).send(r)


