import redis
import json
import pickle
import configs
import base64
import time
import threading
from models import PPOWrap


def encode(object):
    env_pickle = pickle.dumps(object, 0)
    return base64.b64encode(env_pickle).decode()


def decode(object):
    return pickle.loads(base64.b64decode(object.encode()))


class MessageDecoder:
    def __init__(self):
        pass

    def decode(self, message):
        d = json.loads(message['data'])
        msg = d['msg']
        if msg == 'rollout':
            id = d['id']
            policy = decode(d['policy'])
            env_config = decode(d['env_config'])
            return Rollout(id, policy, env_config)

        if msg == 'episode':
            return Episode(d['steps'])

        if msg == 'STOP':
            return Stop()

        if msg == 'STOPALL':
            return StopAll()



class Message:
    def __init__(self):
        self.header = None
        self.content = None
        self.params = None,

    def __equal__(self, string):
        return self.header == string

    def encode(self):
        self.content = f'{{"msg":"{self.header}"}}'

    def send(self, r):
        self.encode()
        r.publish('rollout', self.content)


class Stop(Message):
    def __init__(self):
        super().__init__()
        self.header = 'STOP'


class StopAll(Message):
    def __init__(self):
        super().__init__()
        self.header = 'STOPALL'

    def encode(self):
        self.content = f'{{"msg":"{self.header}"}}'


class Episode(Message):
    def __init__(self, steps):
        super().__init__()
        self.header = 'episode'
        self.steps = int(steps)
        self.params = int(steps),

    def encode(self):
        self.content = f'{{"msg":"{self.header}", "steps":"{self.steps}" }}'


class Rollout(Message):
    def __init__(self, id, policy, env_config):
        super().__init__()
        self.header = 'rollout'
        self.policy = policy
        self.env_config = env_config
        self.id = int(id)
        self.params = int(id), policy, env_config

    def encode(self):
        env_pickle = encode(self.env_config)
        policy_pickle = encode(self.policy)
        self.content = f'{{"msg":"{self.header}", "id":"{self.id}", "policy":"{policy_pickle}", "env_config":"{env_pickle}" }}'


class Server:
    def __init__(self, host='localhost', port=6379, db=0):
        self.r = redis.Redis(host, port, db)
        self.p = self.r.pubsub()
        self.p.subscribe('rollout')
        self.decoder = MessageDecoder()
        self.handler = {'rollout': None, 'episode': None, 'STOP': None, 'STOPALL': None}

    def register(self, msg, callback):
        self.handler[msg] = callback

    def handle(self, message):
        if message['type'] == "message":
            msg = self.decoder.decode(message)
            callback = self.handler[msg.header]
            if callback is not None:
                callback(*msg.params)

    def main(self):
        for message in self.p.listen():
            print(message)
            self.handle(message)


def policy(params):
    print(params)


class RolloutThread(threading.Thread):
    def __init__(self, r, id, policy, env_config):
        super().__init__()
        self.env_config = env_config
        self._stop_event = threading.Event()
        self.r = r

    def run(self):
        for _ in range(20):
            print(f'rolling out {self.env_config.gym_env_string}')
            time.sleep(1)
            Episode(1000).send(self.r)
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
        self.register('rollout', self.rollout)
        self.register('STOP', self.stop)
        self.rollout_thread = None

    def rollout(self, id, policy, env_config):
        self.rollout_thread = RolloutThread(r, id, policy, env_config)
        self.rollout_thread.start()
        print('exited rollout')

    def stop(self, _):
        print('stopping')
        self.rollout_thread.stop()
        self.rollout_thread.join()


class Trainer(Server):
    def __init__(self):
        super().__init__()
        self.register('episode', self.episode)
        self.steps = 0

    def episode(self, steps):
        self.steps += steps
        print(f'got {self.steps} steps')
        if self.steps > 10000:
            print('got data... sending stop')
            Stop().send(self.r)
            time.sleep(10)
            print('training finished')
            self.steps = 0
            Rollout(0, policy_net, env_config).send(self.r)


class GatherThread(threading.Thread):
    def run(self):
        s = Gatherer()
        s.main()


class TrainerThread(threading.Thread):
    def run(self):
        s = Trainer()
        s.main()


if __name__ == '__main__':

    env_config = configs.LunarLander()
    policy_net = PPOWrap(env_config.features, env_config.action_map, env_config.hidden)

    TrainerThread().start()
    GatherThread().start()


    r = redis.Redis()

    Rollout(0, policy_net, env_config).send(r)
    #Episode(1000).send(r)
    #Stop().send(r)
    #StopAll().send(r)


