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
    config.run_id = f'{config.gym_env_string}_{random.randint(0,1000)}'
    StartMessage(gui_uuid, config).send(r)


def stop():
    StopMessage(gui_uuid).send(r)


def demo():
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

    config = configs.LunarLander()

    r = Redis(host=args.redis_host, port=args.redis_port, password=args.redis_password)
    exp_buffer = Db(host=args.redis_host, port=args.redis_port, password=args.redis_password)
    policy_db = PolicyDB(args.postgres_host, args.postgres_password, args.postgres_user, args.postgres_db, args.postgres_port)


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

    pbars, epi_count = gatherer_progressbars(5, config.num_steps_per_rollout)

    # Layout the design of the GUI
    layout = [
        [sg.Text('Please click a button', auto_size_text=True)],
        [sg.ProgressBar(config.num_steps_per_rollout, orientation='h', size=(20, 20), key='trainer'), sg.Text('000000', key='wallclock')],
        pbars,
        epi_count,
        [sg.Button('Start'),
         sg.Button('Stop'),
         sg.Button('Demo'),
         sg.Quit()]
    ]

    # Show the Window to the user
    window = sg.Window('Control Panel').Layout(layout)

    # Event loop. Read buttons, make callbacks
    while True:
        # Read the Window
        event, value = window.Read(timeout=10)
        # Take appropriate action based on button
        if event == 'Start':
            start()
        elif event == 'Stop':
            stop()
        elif event == 'Demo':
            demo()
        elif event == 'Quit' or event is None:
            window.Close()
            break

        handler.checkMessage()

