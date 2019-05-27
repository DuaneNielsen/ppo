import PySimpleGUI as sg
from messages import *
from redis import Redis
import configs
from models import PPOWrap
import uuid
from datetime import datetime
from data import Db
import random
import multiprocessing
from rollout import single_episode
from policy_db import PolicyDB
import gym
import time

# This design pattern simulates button callbacks
# Note that callbacks are NOT a part of the package's interface to the
# caller intentionally.  The underlying implementation actually does use
# tkinter callbacks.  They are simply hidden from the user.

import logging
import duallog

duallog.setup('logs', 'gui')

rollout_time = None


class MicroServiceBuffer:
    def __init__(self):
        self.services = {}
        self.timeout = 10

    def __setitem__(self, key, service):
        service.last_seen = time.time()
        self.services[key] = service

    def del_stale(self):
        for id, service in self.services.items():
            if time.time() - service.last_seen > self.timeout:
                del self.services[id]

    def items(self):
        return self.services.items()

    def values(self):
        return self.services.values()

    def __len__(self):
        return len(self.services)

    def __iter__(self):
        return iter(self.services)

    def __contains__(self, id):
        if type(id) == str:
            return id in self.services
        if type(id) == MicroServiceRec:
            return id.id in self.services

    def __getitem__(self, key):
        return self.services[key]

    def __repr__(self):
        image = ''
        for key, service in self.items():
            image += f'{key} : {service} \n'
        return image


class MicroServiceRec:
    def __init__(self, id, server_info):
        self.id = id
        self.server_info = server_info
        self.last_seen = time.time()

    def __repr__(self):
        return f'id : {self.id}, server_info : {self.server_info}, last_seen: {self.last_seen}, age: {time.time() - self.last_seen}'


class ProgressMap:
    def __init__(self, len, timeout=5):
        self.server_2_slot = {}
        self.slots = ['empty' for _ in range(len)]
        self.epi = [0 for _ in range(len)]
        self.steps = [0 for _ in range(len)]
        self.last_updated = [0 for _ in range(len)]
        self.timeout = timeout

    def update(self, server_id, steps):
        slot = self.get_slot(server_id)
        if slot is not None:
            self.epi[slot] += 1
            self.steps[slot] += steps

    def zero(self, server_id):
        slot = self.get_slot(server_id)
        if slot is not None:
            self.epi[slot] = 0
            self.steps[slot] = 0

    def get_slot(self, server_id):
        if server_id in self.server_2_slot:
            return int(self.server_2_slot[server_id])
        else:
            return None

    def update_bar(self, server_id, num_steps_per_rollout):
        slot = self.get_slot(server_id)
        if slot is not None:
            window.FindElement('gatherer' + str(slot)).UpdateBar(self.steps[slot], max=num_steps_per_rollout)
            window.FindElement('gatherer_epi' + str(slot)).Update(self.epi[slot])
            self.last_updated[slot] = time.time()

    def add(self, server_id):
        for i, slot in enumerate(self.slots):
            if slot == 'empty':
                self.slots[i] = server_id
                self.server_2_slot[server_id] = i
                self.epi[i] = 0
                self.steps[i] = 0
                break

    def free_slot(self, server_id):
        dx = int(self.server_2_slot[server_id])
        self.slots[dx] = 'empty'
        del self.server_2_slot[server_id]

    def clear_old(self):
        now = time.time()
        for slot, last in enumerate(self.last_updated):
            if self.slots[slot] != 'empty' and now - last > self.timeout:
                window.FindElement('gatherer' + str(slot)).UpdateBar(0)
                window.FindElement('gatherer_epi' + str(slot)).Update(0)
                self.free_slot(self.slots[slot])

    def __contains__(self, server_id):
        return server_id in self.server_2_slot


update_count = 0


def episode(msg):
    global update_count
    update_count += 1

    if msg.server_uuid not in progress_map:
        progress_map.add(msg.server_uuid)

    if int(msg.id) == 1:
        progress_map.zero(msg.server_uuid)

    progress_map.update(msg.server_uuid, msg.steps)

    if update_count % 4:
        progress_map.update_bar(msg.server_uuid, msg.num_steps_per_rollout)
        rollout = exp_buffer.latest_rollout(config)
        window.FindElement('trainer').UpdateBar(len(rollout), msg.num_steps_per_rollout)
        window.FindElement('num_steps_per_rollout').Update(msg.num_steps_per_rollout)

def rec_rollout(msg):
    global rollout_time
    rollout_time = datetime.now()


def rec_stop(msg):
    global rollout_time
    if rollout_time != None:
        walltime = datetime.now() - rollout_time
        window.FindElement('wallclock').Update(str(walltime))


def training_progress(msg):
    pass


# The callback functions
def start(config):
    ResetMessage(gui_uuid).send(r)
    StartMessage(gui_uuid, config).send(r)


def stop():
    StopMessage(gui_uuid).send(r)


def demo(run=None):
    if run is None:
        run = policy_db.latest_run()
    record = policy_db.best(run.run).get()
    env = gym.make(record.config_b.gym_env_string)
    policy_net = PPOWrap(record.config_b.features, record.config_b.action_map, record.config_b.hidden)
    policy_net.load_state_dict(record.policy)
    demo = DemoThread(policy_net, record.config_b, env)
    demo.start()


class DemoThread(multiprocessing.Process):
    def __init__(self, policy, env_config, env, num_episodes=1):
        super().__init__()
        self.env_config = env_config
        self.policy = policy.to('cpu').eval()
        self.num_episodes = num_episodes
        self.env = env

    def run(self):
        for episode_number in range(self.num_episodes):
            logging.info(f'starting episode {episode_number} of {self.env_config.gym_env_string}')
            single_episode(self.env, self.env_config, self.policy, render=True)

        self.env.close()
        logging.debug('exiting')


def update_config(value):
    # todo need to read a config into UI first, then send the updated one
    config.num_steps_per_rollout = int(value['num_steps_per_rollout'])
    ConfigUpdateMessage(gui_uuid, config).send(r)


def filter_run(value):
    env_string = value['FilterRun']
    if env_string == 'all':
        runs = policy_db.runs()
    elif env_string == 'latest':
        runs = [policy_db.latest_run().run]
    else:
        runs = policy_db.runs_for_env(env_string)

    best = []
    for run in runs:
        row = policy_db.best(run).get()
        best.append([run, row.stats['ave_reward_episode']])

    best = sorted(best, key=lambda element: element[1], reverse=True)

    window.FindElement('selected_run').Update(best)


def init_run_table():
    runs = [policy_db.latest_run().run]
    best = []
    for run in runs:
        row = policy_db.best(run).get()
        best.append([run.ljust(25), str(row.stats['ave_reward_episode']).ljust(10)])
    return best


def selected_runs(value):
    selected_run = value['selected_run']
    selected_runs = []
    if selected_run is not None:
        for index in selected_run:
            run = window.FindElement('selected_run').Values[index][0]
            selected_runs.append(policy_db.get_latest(run.strip()))

    return selected_runs


def refresh_config_panel():
    for name in config_panel_names:
        window.FindElement(name).Update(config.__dict__[name])


def new_run(value):
    config = config_list[value['selected_config']]
    config.run_id = f'{config.gym_env_string}_{random.randint(0, 1000)}'
    return config


def refresh_progress_panel():
    progress_map.clear_old()


def handle_pong(msg):
    if msg.server_uuid in services:
        services[msg.server_uuid].last_seen = time.time()
    else:
        services[msg.server_uuid] = MicroServiceRec(msg.server_uuid, msg.server_info)

    services.del_stale()
    print(services)


def heartbeat():
    progress_map.clear_old()


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser(description='Start GUI')

    parser.add_argument("-rh", "--redis-host", help='hostname of redis server', dest='redis_host', default='localhost')
    parser.add_argument("-rp", "--redis-port", help='port of redis server', dest='redis_port', default=6379)
    parser.add_argument("-ra", "--redis-password", help='password of redis server', dest='redis_password', default=None)

    parser.add_argument("-ph", "--postgres-host", help='hostname of postgres server', dest='postgres_host',
                        default='localhost')
    parser.add_argument("-pp", "--postgres-port", help='port of postgres server', dest='postgres_port', default=5432)
    parser.add_argument("-pd", "--postgres-db", help='hostname of postgres db', dest='postgres_db',
                        default='policy_db')
    parser.add_argument("-pu", "--postgres-user", help='hostname of postgres user', dest='postgres_user',
                        default='policy_user')
    parser.add_argument("-pa", "--postgres-password", help='password of postgres server', dest='postgres_password',
                        default='password')

    args = parser.parse_args()

    config_list = {
        'CartPole-v0': configs.CartPole(),
        'LunarLander-v2': configs.LunarLander(),
        'Acrobot-v1': configs.Acrobot(),
        'MountainCar-v0': configs.MountainCar()
    }

    r = Redis(host=args.redis_host, port=args.redis_port, password=args.redis_password)
    exp_buffer = Db(host=args.redis_host, port=args.redis_port, password=args.redis_password)
    policy_db = PolicyDB(args.postgres_host, args.postgres_password, args.postgres_user, args.postgres_db,
                         args.postgres_port)

    gui_uuid = uuid.uuid4()

    services = MicroServiceBuffer()
    PingMessage(gui_uuid).send(r)

    gatherers = {}
    gatherers_progress = {}
    gatherers_progress_epi = {}
    next_free_slot = 0

    handler = MessageHandler(r, 'rollout')
    handler.register(EpisodeMessage, episode)
    handler.register(TrainingProgress, training_progress)
    handler.register(StopMessage, rec_stop)
    handler.register(RolloutMessage, rec_rollout)
    handler.register(PongMessage, handle_pong)

    run = policy_db.latest_run()
    config = run.config_b

    """ PROGRESS PANEL """

    def init_progressbars():
        pbars = []
        epi_counters = []
        for i in range(num_bars):
            pbars.append(
                sg.ProgressBar(config.num_steps_per_rollout, orientation='v', size=(20, 20), key='gatherer' + str(i)))
            epi_counters.append(sg.Text('0', size=(3, 2), key='gatherer_epi' + str(i)))

        return pbars, epi_counters

    num_bars = 5
    progress_map = ProgressMap(num_bars)

    pbars, epi_count = init_progressbars()

    progress_panel = [sg.Frame(title='progress', key='progress_panel', layout=
    [
        [sg.Text('Steps per Rollout'), sg.Text('0000000', key='num_steps_per_rollout')],
        [sg.ProgressBar(config.num_steps_per_rollout, orientation='h', size=(20, 20), key='trainer'),
         sg.Text('000000', key='wallclock')],
        pbars,
        epi_count
    ]
                               )]

    """ CONFIG PANEL """


    def config_element(field_name):
        return [sg.Text(field_name, size=(20, 1)),
                sg.In(default_text=str(config.__dict__[field_name]), size=(25, 1),
                      key=field_name)]


    config_panel_names = ['run_id', 'gym_env_string', 'num_steps_per_rollout', 'model_string', 'training_algo',
                          'discount_factor', 'episodes_per_gatherer', 'max_rollout_len', 'policy_reservoir_depth']

    config_panel_elements = []
    for name in config_panel_names:
        config_panel_elements.append(config_element(name))

    config_panel = [sg.Frame(title='config', layout=config_panel_elements)]

    """ NEW RUN PANEL """

    new_run_panel = [sg.Frame(title='new_run', layout=
    [
        [sg.Drop(values=[c for c in config_list], auto_size_text=True, default_value='CartPole-v0',
                 key='selected_config'),
         sg.Button('New Run')]
    ]
                              )]

    """ RUN PANEL """

    env_droplist_value = ['all', 'latest'] + [c for c in config_list]
    env_droplist = sg.Drop(values=env_droplist_value, auto_size_text=True, key='FilterRun', enable_events=True,
                           default_value='latest')
    policy_table = sg.Table(init_run_table(),
                            headings=['run', 'ave reward'], size=(80, 10), key='selected_run')

    run_panel = [sg.Frame(title='runs', layout=
    [
        [env_droplist],
        [policy_table],
        [sg.Button('LoadConfig'),
         sg.Button('Refresh'),
         sg.Button('DemoBest'),
         sg.Button('DemoProgress'),
         sg.Button('Delete')
         ]
    ]
                          )]



    # Layout the design of the GUI
    layout = [
        progress_panel,
        new_run_panel,
        config_panel,
        run_panel,
        [sg.Button('Start'),
         sg.Button('Stop'),
         sg.Button('Demo'),
         sg.Button('UpdateConfig'),
         sg.Quit()]
    ]

    # Show the Window to the user
    window = sg.Window('Control Panel').Layout(layout)


    heartbeat_freq = 2
    time_last_beat = time.time()

    # Event loop. Read buttons, make callbacks
    while True:
        # Read the Window
        event, value = window.Read(timeout=1)

        if time.time() - time_last_beat > heartbeat_freq:
            heartbeat()
            time_last_beat = time.time()

        # Take appropriate action based on button
        if event != '__TIMEOUT__':
            print(event)
        if event == 'Start':
            start(config)
        elif event == 'Stop':
            stop()
        elif event == 'Demo':
            demo()
        elif event == 'UpdateConfig':
            update_config(value)
        elif event == 'FilterRun':
            filter_run(value)
        elif event == 'LoadConfig':
            selected = selected_runs(value)
            if len(selected) == 1:
                config = selected[0].config_b
                refresh_config_panel()
        elif event == 'New Run':
            config = new_run(value)
            refresh_config_panel()
        elif event == 'Refresh':
            filter_run(value)
        elif event == 'DemoBest':
            selected = selected_runs(value)
            if len(selected) == 1:
                demo(selected[0])

        elif event == 'Delete':
            selected = selected_runs(value)
            for run in selected:
                policy_db.delete(run.run)
            filter_run(value)

        elif event == 'Quit' or event is None:
            window.Close()
            break

        handler.checkMessage()
