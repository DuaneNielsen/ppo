import PySimpleGUI as sg
from messages import *
from redis import Redis
import configs
from models import PPOWrap, PPOWrapModel
import uuid
from datetime import datetime
from data import Db
import random
import multiprocessing
from rollout import single_episode
from policy_db import PolicyDB
import gym
import time
from importlib import import_module

# This design pattern simulates button callbacks
# Note that callbacks are NOT a part of the package's interface to the
# caller intentionally.  The underlying implementation actually does use
# tkinter callbacks.  They are simply hidden from the user.

import logging
import duallog

duallog.setup('logs', 'gui')

rollout_time = None


# todo fix status bar performance
# todo type inference of config
# todo add widget for number of demos


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
        rollout = exp_buffer.latest_rollout(config.data.coder)
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


def demo(run=None, best=False, num_episodes=20):
    if run is None:
        run = policy_db.latest_run()
    if best:
        record = policy_db.best(run.run).get()
    else:
        record = policy_db.get_latest(run.run)
    env = gym.make(record.config_b.env.name)
    actor = config.actor.construct()
    actor.load_state_dict(record.actor)
    demo = DemoThread(actor, record.config_b, env, num_episodes=num_episodes)
    demo.start()


class DemoThread(multiprocessing.Process):
    def __init__(self, actor, config, env, num_episodes=1):
        super().__init__()
        self.config = config
        self.actor = actor.to('cpu').eval()
        self.num_episodes = num_episodes
        self.env = env

    def run(self):
        for episode_number in range(self.num_episodes):
            logging.info(f'starting episode {episode_number} of {self.config.env.name}')
            single_episode(self.env, self.config, self.actor, render=True)

        self.env.close()
        logging.debug('exiting')


def update_config(config, values):
    config = cp.read(config, values)
    cp.update(config)


def filter_run(value):
    env_name = value['FilterRun']
    if env_name == 'all':
        runs = policy_db.runs()
    elif env_name == 'latest':
        runs = [policy_db.latest_run().run]
    else:
        runs = policy_db.runs_for_env(env_name)

    best = []
    for run in runs:
        row = policy_db.best(run).get()
        best.append([run, row.stats['ave_reward_episode']])

    best = sorted(best, key=lambda element: element[1], reverse=True)

    window.FindElement('selected_run').Update(best)


def refresh_config_panel():
    cp.update(config)


def new_run(value):
    config = config_list[value['selected_config']]
    config.run_id = f'{config.env.name}_{config.algo.name}_{random.randint(0, 10000)}'
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
        'CartPole-v0': configs.Discrete('CartPole-v0'),
        'LunarLander-v2': configs.Discrete('LunarLander-v2'),
        'Acrobot-v1': configs.Discrete('Acrobot-v1'),
        'MountainCar-v0': configs.Discrete('MountainCar-v0'),
        'HalfCheetah-v1': configs.Continuous('RoboschoolHalfCheetah-v1'),
        'Hopper-v0': configs.Continuous('RoboschoolHopper-v1')
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
    if run is not None:
        config = run.config_b
    else:
        config = configs.default

    """ PROGRESS PANEL """


    def init_progressbars():
        pbars = []
        epi_counters = []
        for i in range(num_bars):
            pbars.append(
                sg.ProgressBar(config.gatherer.num_steps_per_rollout, orientation='v', size=(20, 20),
                               key='gatherer' + str(i)))
            epi_counters.append(sg.Text('0', size=(3, 2), key='gatherer_epi' + str(i)))

        return pbars, epi_counters


    num_bars = 5
    progress_map = ProgressMap(num_bars)

    pbars, epi_count = init_progressbars()

    progress_panel = sg.Frame(title='progress', key='progress_panel', layout=
    [
        [sg.Text('Steps per Rollout'), sg.Text('0000000', key='num_steps_per_rollout')],
        [sg.ProgressBar(config.gatherer.num_steps_per_rollout, orientation='h', size=(20, 20), key='trainer'),
         sg.Text('000000', key='wallclock')],
        pbars,
        epi_count
    ]
                              )

    """ RUN PANEL """

    env_droplist_value = ['all', 'latest'] + [c for c in config_list]
    env_droplist = sg.Drop(values=env_droplist_value, auto_size_text=True, key='FilterRun', enable_events=True,
                           default_value='latest')


    class PolicyTable:
        def __init__(self, policy_db):
            self.best = []
            self.policy_db = policy_db
            latest_run = self.policy_db.latest_run()
            if latest_run is not None:
                runs = [latest_run.run]
                for run in runs:
                    result = self.policy_db.best(run)
                    if len(result) > 0:
                        row = result.get()
                        self.best.append([run.ljust(25), row.stats['ave_reward_episode']])
                    else:
                        self.best.append(['No runs'.ljust(25), ''.ljust(10)])
            else:
                self.best.append(['No runs'.ljust(25), ''.ljust(10)])

        def update(self, value):
            env_name = value['FilterRun']
            if env_name == 'all':
                runs = self.policy_db.runs()
            elif env_name == 'latest':
                latest_run = self.policy_db.latest_run()
                if latest_run is not None:
                    runs = [latest_run.run]
                else:
                    runs = []
            else:
                runs = self.policy_db.runs_for_env(env_name)


            self.best = []
            for run in runs:
                row = self.policy_db.best(run).get()
                self.best.append([run, row.stats['ave_reward_episode']])

            self.best = sorted(self.best, key=lambda element: element[1], reverse=True)

            window.FindElement('selected_run').Update(self.best)

        def layout(self):
            return sg.Table(self.best, headings=['run', 'ave reward'], size=(80, 10), key='selected_run')

        def selected_runs(self, value):
            selected_run = value['selected_run']
            selected_runs = []
            if selected_run is not None:
                for index in selected_run:
                    run = window.FindElement('selected_run').Values[index][0]
                    selected_runs.append(self.policy_db.get_latest(run.strip()))

            return selected_runs


    policy_table = PolicyTable(policy_db)

    run_panel = sg.Frame(title='runs', layout=
    [
        [env_droplist],
        [policy_table.layout()],
        [sg.Button('LoadConfig'),
         sg.Button('Refresh'),
         sg.Button('DemoBest'),
         sg.Button('DemoLatest'),
         sg.Button('Delete')
         ]
    ]
                         )

    """ CONFIG PANEL """


    class ConfigItemWidget:
        def __init__(self, slot_name):
            self.slot_name = slot_name
            self.name = slot_name
            self.value = 'N/A'
            self.row = [sg.Text(slot_name, size=(20, 1)), sg.In(self.value, size=(30, 1), key=slot_name)]
            self.empty = True

        def update(self, name, value, empty=False):
            self.empty = empty
            self.name = name
            self.value = value
            self.row[0].Update(name)
            self.row[1].Update(str(value))

        def read(self):
            t = type(self.value)
            self.value = t(self.row[1].Get())
            return self.name, self.value

        def layout(self):
            return self.row


    class SubPanel:
        def __init__(self, title, rows):
            self.title = title
            self.fields = []
            for row in range(rows):
                self.fields.append(ConfigItemWidget(self.title + str(row)))

            self._layout = None

        def is_serializable(self, obj):
            if isinstance(obj, int) or isinstance(obj, str) or isinstance(obj, float):
                return True
            else:
                return False

        def update(self, config):
            slot = 0
            for name, value in config.__dict__.items():
                if self.is_serializable(value):
                    self.fields[slot].update(name, value)
                    slot += 1

            for slot in range(slot, len(self.fields)):
                self.fields[slot].update('empty', 'N/A', empty=True)

        def read(self, config):
            for name, value in config.__dict__.items():
                if self.is_serializable(value):
                    for field in self.fields:
                        if name == field.name:
                            config.__dict__[name] = field.read()[1]
            return config

        def layout(self):
            self._layout = [configitemwidget.layout() for configitemwidget in self.fields]
            return sg.Frame(title=self.title, layout=self._layout)


    class AlgoPanel(SubPanel):
        def __init__(self, title, rows):
            super().__init__(title, rows)
            self.has_optim = False
            self.optimizer_fields = []
            self.torch_optim_module = import_module('torch.optim')
            for row in range(5):
                self.optimizer_fields.append(ConfigItemWidget(self.title + '_optim' + str(row)))

        def update(self, config):
            super().update(config)
            self.has_optim = 'optimizer' in config.__dict__
            if self.has_optim:
                optimizer = config.__dict__['optimizer']
                self.optimizer_fields[0].update('optim_class', optimizer.clazz.__qualname__)
                slot = 1
                for kwname, kwvalue in optimizer.kwargs.items():
                    if self.is_serializable(kwvalue):
                        self.optimizer_fields[slot].update(kwname, kwvalue)
                        slot += 1
                for slot in range(slot, len(self.optimizer_fields)):
                    self.optimizer_fields[slot].update('empty', 'N/A', empty=True)

        def read(self, config):
            if self.has_optim:
                optim_clazz_name = self.optimizer_fields[0].read()[1]
                optim_clazz = getattr(self.torch_optim_module, optim_clazz_name)
                for field in self.optimizer_fields[1:]:
                    if not field.empty:
                        kw_name, kwvalue = field.read()
                        config.optimizer.kwargs[kw_name] = kwvalue
                config.optimizer = configs.OptimizerConfig(optim_clazz, **config.optimizer.kwargs)
            return super().read(config)

        def layout(self):
            super().layout()
            for configitemwidget in self.optimizer_fields:
                self._layout.append(configitemwidget.row)
            return sg.Frame(title=self.title, layout=self._layout)


    class ConfigPanel:
        def __init__(self):
            self.base = SubPanel('run', 10)
            self.gatherer = SubPanel('gatherer', 10)
            self.env = SubPanel('env', 10)
            self.algo = AlgoPanel('algo', 10)

        def update(self, config):
            self.base.update(config)
            self.env.update(config.env)
            self.gatherer.update(config.gatherer)
            self.algo.update(config.algo)

        def read(self, config, values):
            from copy import copy
            orig_config = copy(config)
            self.base.read(config)
            self.gatherer.read(config.gatherer)
            self.algo.read(config.algo)
            self.env.read(config.gatherer)
            return config

        def layout(self):
            line1 = []
            line2 = []
            # line1.append(sg.Column([[self.base.frame()], [self.env.frame()]]))
            line1.append(self.base.layout())
            line1.append(self.gatherer.layout())
            line1.append(self.env.layout())
            line1.append(self.algo.layout())
            return line1, line2


    cp = ConfigPanel()

    view = config.view()

    config_panel_l1, config_panel_l2 = cp.layout()

    """ NEW RUN PANEL """

    new_run_panel = [sg.Frame(title='new_run', layout=
    [
        [sg.Drop(values=[c for c in config_list], auto_size_text=True, default_value='CartPole-v0',
                 key='selected_config'),
         sg.Button('New Run')]
    ]
                              )]

    # Layout the design of the GUI
    layout = [
        [progress_panel, run_panel],
        new_run_panel,
        config_panel_l1,
        config_panel_l2,
        [sg.Button('Start'),
         sg.Button('Stop'),
         sg.Button('Demo'),
         sg.Button('UpdateConfig'),
         sg.Quit()]
    ]

    # Show the Window to the user
    window = sg.Window('Control Panel').Layout(layout)
    window.Finalize()

    cp.update(config)

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
            update_config(config, value)
            start(config)
        elif event == 'Stop':
            stop()
        elif event == 'Demo':
            demo()
        elif event == 'UpdateConfig':
            update_config(config, value)
            ConfigUpdateMessage(gui_uuid, config).send(r)
        elif event == 'FilterRun':
            policy_table.update(value)
        elif event == 'LoadConfig':
            selected = policy_table.selected_runs(value)
            if len(selected) == 1:
                config = selected[0].config_b
                refresh_config_panel()
        elif event == 'New Run':
            config = new_run(value)
            refresh_config_panel()
        elif event == 'Refresh':
            policy_table.update(value)
        elif event == 'DemoBest':
            selected = policy_table.selected_runs(value)
            if len(selected) == 1:
                demo(selected[0], best=True)

        elif event == 'DemoLatest':
            selected = policy_table.selected_runs(value)
            if len(selected) == 1:
                demo(selected[0])

        elif event == 'Delete':
            selected = policy_table.selected_runs(value)
            for run in selected:
                policy_db.delete(run.run)
            filter_run(value)

        elif event == 'Quit' or event is None:
            window.Close()
            break

        handler.checkMessage()
