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

# This design pattern simulates button callbacks
# Note that callbacks are NOT a part of the package's interface to the
# caller intentionally.  The underlying implementation actually does use
# tkinter callbacks.  They are simply hidden from the user.

import logging
import duallog

duallog.setup('logs', 'gui')

rollout_time = None


def episode(msg):
    global next_free_slot

    # add a new slot if needed
    if msg.server_uuid not in gatherers:
        gatherers[msg.server_uuid] = str(next_free_slot)
        gatherers_progress[msg.server_uuid] = 0
        gatherers_progress_epi[msg.server_uuid] = 0
        next_free_slot += 1

    if int(msg.id) == 1:
        gatherers_progress[msg.server_uuid] = 0
        gatherers_progress_epi[msg.server_uuid] = 0

    if msg.server_uuid in gatherers:
        gatherers_progress[msg.server_uuid] += int(msg.steps)
        gatherers_progress_epi[msg.server_uuid] += 1

    window.FindElement('gatherer' + gatherers[msg.server_uuid]).UpdateBar(gatherers_progress[msg.server_uuid])
    window.FindElement('gatherer_epi' + gatherers[msg.server_uuid]).Update(gatherers_progress_epi[msg.server_uuid])

    rollout = exp_buffer.latest_rollout(config)
    window.FindElement('trainer').UpdateBar(len(rollout))


def rec_rollout(msg):
    global rollout_time
    rollout_time = datetime.now()


def rec_stop(msg):
    global rollout_time
    if rollout_time != None:
        walltime = datetime.now() - rollout_time
        window.FindElement('wallclock').Update(str(walltime))


def gatherer_progressbars(number, max_episodes):
    pbars = []
    epi_counters = []
    for i in range(number):
        pbars.append(sg.ProgressBar(max_episodes, orientation='v', size=(20, 20), key='gatherer' + str(i)))
        epi_counters.append(sg.Text('0', size=(3, 2), key='gatherer_epi' + str(i)))

    return pbars, epi_counters


def training_progress(msg):
    pass


# The callback functions
def start():
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
        best.append([run, row.stats['ave_reward_episode']])
    return best


def selected_runs(value):
    selected_run = value['selected_run']
    selected_runs = []
    if selected_run is not None:
        for index in selected_run:
            run = window.FindElement('selected_run').Values[index][0]
            selected_runs.append(policy_db.get_latest(run))

    return selected_runs


def refresh_config_panel():
    for name in config_panel_names:
        window.FindElement(name).Update(config.__dict__[name])


def new_run(value):
    config = config_list[value['selected_config']]
    config.run_id = f'{config.gym_env_string}_{random.randint(0, 1000)}'
    return config


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
        'Acrobot-v1': configs.Acrobot()
    }

    r = Redis(host=args.redis_host, port=args.redis_port, password=args.redis_password)
    exp_buffer = Db(host=args.redis_host, port=args.redis_port, password=args.redis_password)
    policy_db = PolicyDB(args.postgres_host, args.postgres_password, args.postgres_user, args.postgres_db,
                         args.postgres_port)

    gui_uuid = uuid.uuid4()

    gatherers = {}
    gatherers_progress = {}
    gatherers_progress_epi = {}
    next_free_slot = 0

    handler = MessageHandler(r, 'rollout')
    handler.register(EpisodeMessage, episode)
    handler.register(TrainingProgress, training_progress)
    handler.register(StopMessage, rec_stop)
    handler.register(RolloutMessage, rec_rollout)

    run = policy_db.latest_run()
    config = run.config_b

    """ PROGRESS BARS """

    pbars, epi_count = gatherer_progressbars(5, config.num_steps_per_rollout)

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

    """ PROGRESS PANEL """

    progress_panel = [sg.Frame(title='progress', layout=
    [
        [sg.ProgressBar(config.num_steps_per_rollout, orientation='h', size=(20, 20), key='trainer'),
         sg.Text('000000', key='wallclock')],
        pbars,
        epi_count
    ]
    )]

    """ NEW RUN PANEL """

    new_run_panel = [sg.Frame(title='new_run', layout=
                         [
                             [sg.Drop(values=[c for c in config_list], auto_size_text=True, default_value='CartPole-v0', key='selected_config'),
                              sg.Button('New Run')]
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

    # Event loop. Read buttons, make callbacks
    while True:
        # Read the Window
        event, value = window.Read(timeout=10)
        # Take appropriate action based on button
        if event != '__TIMEOUT__':
            print(event)
        if event == 'Start':
            selected_config = value['selected_config']
            config = config_list[selected_config]
            start()
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
