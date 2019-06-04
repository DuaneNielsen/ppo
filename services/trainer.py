import logging
from services.server import Server
from data import Db, RolloutDatasetBase
from messages import TrainMessage, TrainCompleteMessage
from ppo_clip_discrete import train_policy, train_ppo_continuous

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s-%(module)s-%(message)s', level=logging.INFO)
logging.getLogger('peewee').setLevel(logging.INFO)

class Trainer(Server):
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0, redis_password=None):
        super().__init__(redis_host, redis_port, redis_db, redis_password)
        self.handler.register(TrainMessage, self.handle_train)
        self.exp_buffer = Db(host=redis_host, port=redis_port, db=redis_db, password=redis_password)

    def handle_train(self, msg):
        rollout = self.exp_buffer.latest_rollout(msg.config)
        assert len(rollout) != 0
        dataset = RolloutDatasetBase(msg.config, rollout)
        policy = msg.policy

        logging.debug('started training')
        if msg.config.continuous:
            train_ppo_continuous(policy, dataset, msg.config)
        else:
            train_policy(policy, dataset, msg.config)
        logging.debug('training complete')
        TrainCompleteMessage(self.id, policy, msg.config).send(self.r)