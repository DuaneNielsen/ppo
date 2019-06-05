import logging
from services.server import Server
from data import Db, RolloutDatasetBase
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

        if msg.config.training_algo == 'ppo':
            trainer = algos.PurePPOClip()
        elif msg.config.training_algo == 'td_zero':
            trainer = algos.OneStepTD()
        else:
            raise NoTrainingAlgo

        exp_buffer = self.exp_buffer.latest_rollout(msg.config)
        assert len(exp_buffer) != 0
        policy = msg.policy

        logger.info('started training')
        trainer(policy, exp_buffer, msg.config)

        logging.info('training complete')
        TrainCompleteMessage(self.id, policy, msg.config).send(self.r)