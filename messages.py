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
    def __init__(self, server_uuid, id, steps):
        super().__init__(server_uuid)
        self.id = id
        self.steps = int(steps)

    def encode(self):
        self.content = f'{{ {self._header_content}, "id":"{self.id}", "steps":"{self.steps}" }}'

    @classmethod
    def header(cls):
        return 'episode'

    @classmethod
    def decode(cls, encoded):
        return EpisodeMessage(encoded['server_uuid'], encoded['id'], encoded['steps'])


class RolloutMessage(Message):
    def __init__(self, server_uuid, id, policy, env_config):
        super().__init__(server_uuid)
        self.policy = policy
        self.env_config = env_config
        self.id = int(id)

    def encode(self):
        env_pickle = encode(self.env_config)
        policy_pickle = encode(self.policy)
        self.content = f'{{ {self._header_content}, "id":"{self.id}", "policy":"{policy_pickle}", "env_config":"{env_pickle}" }}'

    @classmethod
    def header(cls):
        return 'rollout'

    @classmethod
    def decode(cls, d):
        server_uuid = d['server_uuid']
        id = d['id']
        policy = decode(d['policy'])
        env_config = decode(d['env_config'])
        return RolloutMessage(server_uuid, id, policy, env_config)


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