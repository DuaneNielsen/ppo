from services.server import Server
from messages import *
import redis
import pytest


@pytest.fixture
def r():
    return redis.Redis('localhost')


class ListenTestServer(Server):
    def __init__(self, r):
        super().__init__(redis_client=r)
        self.handler.register(TrainCompleteMessage, self.handle_train_complete)

    def handle_train_complete(self, msg):
        print(msg)




