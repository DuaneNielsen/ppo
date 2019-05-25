import base64
import json
import pickle
import logging


def encode(object):
    env_pickle = pickle.dumps(object, 0)
    return base64.b64encode(env_pickle).decode()


def decode(object):
    return pickle.loads(base64.b64decode(object.encode()))


class MessageDecoder:
    def __init__(self):
        self.lookup = {}
        self.register(RolloutMessage)
        self.register(EpisodeMessage)
        self.register(StopMessage)
        self.register(StopAllMessage)
        self.register(StoppedMessage)
        self.register(ResetMessage)
        self.register(TrainingProgress)
        self.register(StartMessage)
        self.register(TrainMessage)
        self.register(TrainCompleteMessage)
        self.register(ExitMessage)
        self.register(ConfigUpdateMessage)
        self.register(PingMessage)
        self.register(PongMessage)

    def register(self, message_class):
        """ Registers a message class's decode in a lookup table"""
        self.lookup[message_class.header()] = message_class.decode

    def decode(self, message):
        d = json.loads(message['data'])
        msg = d['msg']

        # lookup the decode method and pass it the message
        if msg in self.lookup:
            return self.lookup[msg](d)
        else:
            raise Exception


class Message:
    def __init__(self, server_uuid):
        self.content = None
        self._header = self.header()
        self.server_uuid = server_uuid
        self._header_content = f'"msg":"{self._header}", "server_uuid": "{self.server_uuid}"'

    def encode(self):
        self.content = f'{{{self._header_content}}}'

    def send(self, r):
        self.encode()
        r.publish('rollout', self.content)

    @classmethod
    def header(cls):
        return cls.header()

    @classmethod
    def decode(cls, d):
        return cls(d['server_uuid'])


class StartMessage(Message):
    def __init__(self, server_uuid, config):
        super().__init__(server_uuid)
        self.config = config

    def encode(self):
        env_pickle = encode(self.config)
        self.content = \
            (
                f'{{'
                f'{self._header_content},'
                f'"env_config": "{env_pickle}"'
                f'}}'
            )

    @classmethod
    def header(cls):
        return 'START'

    @classmethod
    def decode(cls, d):
        server_uuid = d['server_uuid']
        config = decode(d['env_config'])
        return StartMessage(server_uuid, config)


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
    def __init__(self, server__uuid):
        super().__init__(server__uuid)

    @classmethod
    def header(cls):
        return 'STOPPED'


class ExitMessage(Message):
    def __init__(self, server__uuid):
        super().__init__(server__uuid)

    @classmethod
    def header(cls):
        return 'EXIT'


class PingMessage(Message):
    def __init__(self, server__uuid):
        super().__init__(server__uuid)

    @classmethod
    def header(cls):
        return 'PING'


class PongMessage(Message):
    def __init__(self, server__uuid, server_info):
        super().__init__(server__uuid)
        self.server_info = server_info

    def encode(self):
        self.content = \
            (
                f'{{ '
                f'{self._header_content}, '
                f'"server_info":"{self.server_info}"'
                f'}}'
            )

    @classmethod
    def header(cls):
        return 'PONG'

    @classmethod
    def decode(cls, d):
        server_uuid = d['server_uuid']
        server_info = d['server_info']
        return PongMessage(server_uuid, server_info)


class TrainMessage(Message):
    def __init__(self, server_uuid, policy, config):
        super().__init__(server_uuid)
        self.policy = policy
        self.config = config

    def encode(self):
        env_pickle = encode(self.config)
        policy_pickle = encode(self.policy)
        self.content = \
            (
                f'{{ '
                f'{self._header_content}, '
                f'"policy":"{policy_pickle}", '
                f'"env_config":"{env_pickle}" '
                f'}}'
            )

    @classmethod
    def header(cls):
        return 'train'

    @classmethod
    def decode(cls, d):
        server_uuid = d['server_uuid']
        policy = decode(d['policy'])
        config = decode(d['env_config'])
        return TrainMessage(server_uuid, policy, config)


class ConfigUpdateMessage(Message):
    def __init__(self, server_uuid, config):
        super().__init__(server_uuid)
        self.config = config

    def encode(self):
        config_pickle = encode(self.config)
        self.content = \
            (
                f'{{ '
                f'{self._header_content}, '
                f'"config":"{config_pickle}" '
                f'}}'
            )

    @classmethod
    def header(cls):
        return 'config_update'

    @classmethod
    def decode(cls, d):
        server_uuid = d['server_uuid']
        config = decode(d['config'])
        return ConfigUpdateMessage(server_uuid, config)


class TrainCompleteMessage(Message):
    def __init__(self, server_uuid, policy, config):
        super().__init__(server_uuid)
        self.policy = policy
        self.config = config

    def encode(self):
        env_pickle = encode(self.config)
        policy_pickle = encode(self.policy)
        self.content = \
            (
                f'{{ '
                f'{self._header_content}, '
                f'"policy":"{policy_pickle}", '
                f'"env_config":"{env_pickle}" '
                f'}}'
            )

    @classmethod
    def header(cls):
        return 'train_complete'

    @classmethod
    def decode(cls, d):
        server_uuid = d['server_uuid']
        policy = decode(d['policy'])
        config = decode(d['env_config'])
        return TrainCompleteMessage(server_uuid, policy, config)


class TrainingProgress(Message):
    def __init__(self, server_uuid, steps):
        super().__init__(server_uuid)
        self.steps = steps

    @classmethod
    def header(cls):
        return 'training_progress'

    def encode(self):
        self.content = f'{{ {self._header_content}, "steps":{self.steps} }}'

    @classmethod
    def decode(cls, encoded):
        return TrainingProgress(encoded['server_uuid'], encoded['steps'])


class EpisodeMessage(Message):
    def __init__(self, server_uuid, id, steps, total_reward):
        super().__init__(server_uuid)
        self.id = id
        self.steps = int(steps)
        self.total_reward = float(total_reward)

    def encode(self):
        self.content = f'{{ {self._header_content}, "id":"{self.id}", "steps":"{self.steps}", "total_reward":"{self.total_reward}"}}'

    @classmethod
    def header(cls):
        return 'episode'

    @classmethod
    def decode(cls, encoded):
        return EpisodeMessage(encoded['server_uuid'], encoded['id'], encoded['steps'], encoded['total_reward'])


class RolloutMessage(Message):
    def __init__(self, server_uuid, rollout_id, policy, config, episodes):
        super().__init__(server_uuid)
        self.policy = policy
        self.rollout_id = int(rollout_id)
        self.config = config
        self.episodes = episodes

    def encode(self):
        env_pickle = encode(self.config)
        policy_pickle = encode(self.policy)
        self.content = \
            (
                f'{{ '
                f'{self._header_content}, '
                f'"rollout_id":"{self.rollout_id}", '
                f'"policy":"{policy_pickle}", '
                f'"env_config":"{env_pickle}", '
                f'"episodes":"{self.episodes}" '
                f'}}'
            )

    @classmethod
    def header(cls):
        return 'rollout'

    @classmethod
    def decode(cls, d):
        server_uuid = d['server_uuid']
        id = d['rollout_id']
        policy = decode(d['policy'])
        config = decode(d['env_config'])
        episodes = d['episodes']
        return RolloutMessage(server_uuid, id, policy, config, episodes)


class MessageHandler:
    def __init__(self, redis, channel):
        self.p = redis.pubsub()
        self.p.subscribe(channel)
        self.decoder = MessageDecoder()
        self.handler = {}

    def register(self, msg, callback):
        self.handler[msg.header()] = callback

    def handle(self, message):
        if message['type'] == "message":
            msg = self.decoder.decode(message)
            if msg.header() in self.handler:
                callback = self.handler[msg.header()]
                if callback is not None:
                    callback(msg)

    def listen(self):
        """Blocking call for main loop"""
        for message in self.p.listen():
            logging.debug(message)
            self.handle(message)

    def checkMessage(self):
        """"Non blocking call for busy wait loop"""
        msg = self.p.get_message()
        if msg is not None:
            self.handle(msg)
