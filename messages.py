import jsonpickle
import pickle
import base64
import torch


class ModuleHandler(jsonpickle.handlers.BaseHandler):
    def flatten(self, obj, data):
        env_pickle = pickle.dumps(obj, 0)
        data['ModuleHander_bytestring'] = base64.b64encode(env_pickle).decode()
        return data

    def restore(self, data):
        encoded = data['ModuleHander_bytestring']
        decoded = base64.b64decode(encoded)
        return pickle.loads(decoded)


ModuleHandler.handles(torch.Tensor)


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
        return cls.__name__

    def send(self, r):
        r.publish('rollout', self.encode())


class StartMessage(Message):
    def __init__(self, server_uuid, config):
        super().__init__(server_uuid)
        self.config = config


class StopMessage(Message):
    def __init__(self, server_uuid):
        super().__init__(server_uuid)


class StopAllMessage(Message):
    def __init__(self, server_uuid):
        super().__init__(server_uuid)


class ResetMessage(Message):
    def __init__(self, server_uuid):
        super().__init__(server_uuid)


class StoppedMessage(Message):
    def __init__(self, server_uuid):
        super().__init__(server_uuid)


class ExitMessage(Message):
    def __init__(self, server_uuid):
        super().__init__(server_uuid)


class PingMessage(Message):
    def __init__(self, server_uuid):
        super().__init__(server_uuid)


class PongMessage(Message):
    def __init__(self, server__uuid, server_info):
        super().__init__(server__uuid)
        self.server_info = server_info


class TrainMessage(Message):
    def __init__(self, server_uuid, policy, config):
        super().__init__(server_uuid)
        self.policy = policy
        self.config = config


class StartMonitoringMessage(Message):
    def __init__(self, server_uuid, run):
        super().__init__(server_uuid)
        self.run = run


class ConfigUpdateMessage(Message):
    def __init__(self, server_uuid, config):
        super().__init__(server_uuid)
        self.config = config


class TrainCompleteMessage(Message):
    def __init__(self, server_uuid, policy, config):
        super().__init__(server_uuid)
        self.policy = policy
        self.config = config


class TrainingProgress(Message):
    def __init__(self, server_uuid, steps):
        super().__init__(server_uuid)
        self.steps = steps


class EpisodeMessage(Message):
    def __init__(self, server_uuid, run, episode_number, steps, total_reward, num_steps_rollout):
        super().__init__(server_uuid)
        self.run = run
        self.id = episode_number
        self.steps = int(steps)
        self.total_reward = float(total_reward)
        self.num_steps_per_rollout = int(num_steps_rollout)
        self.monitor = {}


class RolloutMessage(Message):
    def __init__(self, server_uuid, run, rollout_id, policy, config, episodes):
        super().__init__(server_uuid)
        self.run = run
        self.policy = policy
        self.rollout_id = int(rollout_id)
        self.config = config
        self.episodes = episodes


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
