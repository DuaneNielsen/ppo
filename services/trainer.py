import logging
from services.server import Server
from data import Db
from messages import TrainMessage, TrainCompleteMessage
import algos

logger = logging.getLogger(__name__)


class NoTrainingAlgo(Exception):
    pass


class Trainer(Server):
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0, redis_password=None):
        super().__init__(redis_host, redis_port, redis_db, redis_password)
        self.handler.register(TrainMessage, self.handle_train)
        self.exp_buffer = Db(host=redis_host, port=redis_port, db=redis_db, password=redis_password)

        logger.info('Init Complete')

    def handle_train(self, msg):

        trainer = msg.config.algo.construct()

        exp_buffer = self.exp_buffer.latest_rollout(msg.config)
        assert len(exp_buffer) != 0

        logger.info('started training')
        actor, critic = trainer(msg.critic, exp_buffer, msg.config)

        logging.info('training complete')
        TrainCompleteMessage(self.id, actor, critic, msg.config).send(self.r)