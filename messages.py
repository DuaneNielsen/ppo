import jsonpickle
import pickle
import base64


class ModuleHandler(jsonpickle.handlers.BaseHandler):
    def flatten(self, obj, data):
        env_pickle = pickle.dumps(obj, 0)
        data['model_bytes'] = base64.b64encode(env_pickle).decode()
        return data

    def restore(self, data):
        encoded = data['model_bytes']
        decoded = base64.b64decode(encoded)
        return pickle.loads(decoded)


class JSONPickleCoder:
    @staticmethod
    def encode(msg):
        return jsonpickle.encode(msg)

    @staticmethod
    def decode(encoded):
        return jsonpickle.decode(encoded)


class Message:
    def __init__(self, server_uuid):
        self._header = self.header()
        self.server_uuid = server_uuid

    def encode(self):
        return JSONPickleCoder.encode(self)

    def decode(self, encoded):
        return JSONPickleCoder.decode(encoded)

    @classmethod
    def header(cls):
        return ''

    def send(self, r):
        r.publish('rollout', self.encode())


class StartMessage(Message):
    def __init__(self, server_uuid, config):
        super().__init__(server_uuid)
        self.config = config

    @classmethod
    def header(cls):
        return 'START'


class StopMessage(Message):
    def __init__(self, server_uuid):
        super().__init__(server_uuid)

    @classmethod
    def header(cls):
        return 'STOP'


class StopAllMessage(Message):
    def __init__(self, server_uuid):
        super().__init__(server_uuid)

    @classmethod
    def header(cls):
        return 'STOPALL'


class ResetMessage(Message):
    def __init__(self, server_uuid):
        super().__init__(server_uuid)

    @classmethod
    def header(cls):
        return 'RESET'


class StoppedMessage(Message):
    def __init__(self, server_uuid):
        super().__init__(server_uuid)

    @classmethod
    def header(cls):
        return 'STOPPED'


class ExitMessage(Message):
    def __init__(self, server_uuid):
        super().__init__(server_uuid)

    @classmethod
    def header(cls):
        return 'EXIT'


class PingMessage(Message):
    def __init__(self, server_uuid):
        super().__init__(server_uuid)

    @classmethod
    def header(cls):
        return 'PING'


class PongMessage(Message):
    def __init__(self, server__uuid, server_info):
        super().__init__('PONG', server__uuid)
        self.server_info = server_info


class TrainMessage(Message):
    def __init__(self, server_uuid, policy, config):
        super().__init__(server_uuid)
        self.policy = policy
        self.config = config

    @classmethod
    def header(cls):
        return 'train'


class ConfigUpdateMessage(Message):
    def __init__(self, server_uuid, config):
        super().__init__(server_uuid)
        self.config = config

    @classmethod
    def header(cls):
        return 'config_update'


class TrainCompleteMessage(Message):
    def __init__(self, server_uuid, policy, config):
        super().__init__(server_uuid)
        self.policy = policy
        self.config = config

    @classmethod
    def header(cls):
        return 'train_complete'


class TrainingProgress(Message):
    def __init__(self, server_uuid, steps):
        super().__init__(server_uuid)
        self.steps = steps

    @classmethod
    def header(cls):
        return 'training_progress'


class EpisodeMessage(Message):
    def __init__(self, server_uuid, id, steps, total_reward, num_steps_rollout):
        super().__init__(server_uuid)
        self.id = id
        self.steps = int(steps)
        self.total_reward = float(total_reward)
        self.num_steps_per_rollout = int(num_steps_rollout)
        self.monitor = {}

    @classmethod
    def header(cls):
        return 'episode'


class RolloutMessage(Message):
    def __init__(self, server_uuid, rollout_id, policy, config, episodes):
        super().__init__(server_uuid)
        self.policy = policy
        self.rollout_id = int(rollout_id)
        self.config = config
        self.episodes = episodes

    @classmethod
    def header(cls):
        return 'rollout'


class MessageHandler:
    def __init__(self, redis, channel):
        self.p = redis.pubsub()
        self.p.subscribe(channel)
        self.handler = {}

    def register(self, msg, callback):
        self.handler[msg.header] = callback

    def handle(self, message):
        if message['type'] == "message":
            msg = JSONPickleCoder.decode(message['data'])
            if msg.header in self.handler:
                callback = self.handler[msg.header]
                if callback is not None:
                    callback(msg)

    def listen(self):
        """Blocking call for main loop"""
        for message in self.p.listen():
            self.handle(message)

    def checkMessage(self):
        """"Non blocking call for busy wait loop"""
        msg = self.p.get_message()
        if msg is not None:
            self.handle(msg)
