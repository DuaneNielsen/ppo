import logging
import traceback
import uuid
from time import sleep

import redis

from messages import MessageHandler, ExitMessage, PingMessage, PongMessage


class Server:
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0, redis_password=None, redis_client=None):
        self.id = uuid.uuid4()

        if redis_client is not None:
            self.r = redis_client
        else:
            self.r = redis.Redis(host=redis_host, port=redis_port, db=redis_db, password=redis_password)

        self.handler = MessageHandler(self.r, 'rollout')
        self.handler.register(ExitMessage, self.exit)
        self.handler.register(PingMessage, self.handle_ping)
        self.retry_count = 0

    def main(self):
        while self.retry_count < 10:
            try:
                self.retry_count = 0
                self.handler.listen()
            except redis.exceptions.ConnectionError as e:
                logging.error(e)
                self.retry_count += 1
                sleep(self.retry_count)
                continue
            except Exception as e:
                logging.error(e)
                logging.debug(traceback.format_exc())
                self.retry_count += 1
                continue

    def exit(self, msg):
        raise SystemExit

    def handle_ping(self, msg):
        PongMessage(self.id, type(self).__name__).send(self.r)