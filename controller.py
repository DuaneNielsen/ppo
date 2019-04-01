import redis
import json
import pickle
import configs
import base64
import time
import threading
from models import PPOWrap
import uuid


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
            print(message)
            self.handle(message)

    def checkMessage(self):
        """"Non blocking call for busy wait loop"""
        msg = self.p.get_message()
        if msg is not None:
            self.handle(msg)



class Server:
    def __init__(self, host='localhost', port=6379, db=0):
        self.id = uuid.uuid4()
        self.r = redis.Redis(host, port, db)
        self.handler = MessageHandler(self.r, 'rollout')
        self.stopped = False
        self.handler.register(ResetMessage, self.reset)
        self.handler.register(StopAllMessage, self.stopAll)

    def main(self):
        self.handler.listen()

    def reset(self, _):
        self.stopped = False

    def stopAll(self, msg):
        self.stopped = True
        StoppedMessage(self.id).send(self.r)


def policy(params):
    print(params)


class RolloutThread(threading.Thread):
    def __init__(self, r, server_uuid, id, policy, env_config):
        super().__init__()
        self.env_config = env_config
        self._stop_event = threading.Event()
        self.r = r
        self.server_uuid = server_uuid

    def run(self):
        for episode in range(20):
            print(f'rolling out {self.env_config.gym_env_string}')
            time.sleep(1)
            EpisodeMessage(self.server_uuid, episode, 1000).send(self.r)
            if self.stopped():
                print('thread stopped, exiting')
                break

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


class Gatherer(Server):
    def __init__(self):
        super().__init__()
        self.handler.register(RolloutMessage, self.rollout)
        self.handler.register(StopMessage, self.stop)
        self.rollout_thread = None

    def rollout(self, msg):
        if not self.stopped:
            self.rollout_thread = RolloutThread(self.r, self.id, msg.id, msg.policy, msg.env_config)
            self.rollout_thread.start()

    def _stop(self):
        if self.rollout_thread is not None:
            self.rollout_thread.stop()
            self.rollout_thread.join()

    def stop(self, msg):
        self._stop()

    def stopAll(self, msg):
        self._stop()
        super().stopAll(msg)


class Trainer(Server):
    def __init__(self, env_config):
        super().__init__()
        self.handler.register(EpisodeMessage, self.episode)
        self.steps = 0
        self.env_config = env_config

    def episode(self, msg):
        if not self.stopped:
            self.steps += msg.steps
            print(f'got {self.steps} steps')
            if self.steps > 10000:
                print('got data... sending stop')
                StopMessage(self.id).send(self.r)
                time.sleep(5)
                policy_net = PPOWrap(self.env_config.features, self.env_config.action_map, self.env_config.hidden)
                print('training finished')
                self.steps = 0
                RolloutMessage(self.id, 0, policy_net, self.env_config).send(self.r)

    def stopAll(self, msg):
        super().stopAll(msg)
        self.steps = 0


class GatherThread(threading.Thread):
    def run(self):
        s = Gatherer()
        s.main()


class TrainerThread(threading.Thread):
    def run(self):
        s = Trainer(env_config)
        s.main()


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Start server.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-t", "--trainer", help="start a training instance",
                       action="store_true")
    group.add_argument("-g", "--gatherer", help="start a gathering instance",
                       action="store_true")
    group.add_argument("--start", help="start training",
                       action="store_true")
    group.add_argument("--stopall", help="stop training",
                       action="store_true")
    args = parser.parse_args()

    env_config = configs.LunarLander()
    policy_net = PPOWrap(env_config.features, env_config.action_map, env_config.hidden)

    r = redis.Redis()

    if args.trainer:
        trainer = Trainer(env_config)
        trainer.main()
    elif args.gatherer:
        getherer = Gatherer()
        getherer.main()
    elif args.start:
        ResetMessage().send(r)
        RolloutMessage(0, policy_net, env_config).send(r)
    elif args.stopall:
        StopAllMessage().send(r)






    #
    # t1 = TrainerThread()
    # g1 = GatherThread()
    #
    # t1.start()
    # g1.start()
    #
    # r = redis.Redis()
    #
    # RolloutMessage(0, policy_net, env_config).send(r)
    #
    # time.sleep(20)
    #
    # StopAllMessage().send(r)
    #
    # time.sleep(3)
