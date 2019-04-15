import PySimpleGUI as sg
from messages import StopAllMessage, ResetMessage, EpisodeMessage, RolloutMessage, MessageHandler, \
    TrainingProgress
from redis import Redis
import configs
from models import PPOWrap
import uuid

# This design pattern simulates button callbacks
# Note that callbacks are NOT a part of the package's interface to the
# caller intentionally.  The underlying implementation actually does use
# tkinter callbacks.  They are simply hidden from the user.

import logging
import duallog

duallog.setup('logs', 'gui')

def episode(msg):
    global next_free_slot

    if msg.server_uuid not in gatherers:
        gatherers[msg.server_uuid] = 'gatherer' + str(next_free_slot)
        gatherers_progress[msg.server_uuid] = 0
        next_free_slot += 1

    if int(msg.id) == 1:
        gatherers_progress[msg.server_uuid] = 0
    else:
        gatherers_progress[msg.server_uuid] += int(msg.steps)
    window.FindElement(gatherers[msg.server_uuid]).UpdateBar(gatherers_progress[msg.server_uuid])




def gatherer_progressbars(number, max_episodes):
    pbars = []
    for i in range(number):
        pbars.append(sg.ProgressBar(max_episodes, orientation='v', size=(20, 20), key='gatherer' + str(i)))
    return pbars


def training_progress(msg):
    window.FindElement('trainer').UpdateBar(int(msg.steps))




# The callback functions
def start():
    ResetMessage(gui_uuid).send(r)
    RolloutMessage(gui_uuid, 0, policy_net, config).send(r)


def stop():
    StopAllMessage(gui_uuid).send(r)


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser(description='Start server.')

    parser.add_argument("-rh", "--redis-host", help='hostname of redis server', dest='redis_host')
    parser.add_argument("-rp", "--redis-port", help='hostname of redis server', dest='redis_port')
    parser.add_argument("-ra", "--redis-password", help='hostname of redis server', dest='redis_password')
    args = parser.parse_args()

    config = configs.LunarLander()
    config.redis_host = args.redis_host if args.redis_host is not None else config.redis_host
    config.redis_port = args.redis_port if args.redis_port is not None else config.redis_port
    config.redis_password = args.redis_password if args.redis_password is not None else config.redis_password

    r = Redis(host=config.redis_host, port=config.redis_port, password=config.redis_password)

    policy_net = PPOWrap(config.features, config.action_map, config.hidden)
    gui_uuid = uuid.uuid4()

    gatherers = {}
    gatherers_progress = {}
    next_free_slot = 0

    handler = MessageHandler(r, 'rollout')
    handler.register(EpisodeMessage, episode)
    handler.register(TrainingProgress, training_progress)

    # Layout the design of the GUI
    layout = [
        [sg.Text('Please click a button', auto_size_text=True)],
        [sg.ProgressBar(10000, orientation='h', size=(20, 20), key='trainer')],
        gatherer_progressbars(5, 10000),
        [sg.Button('Start'),
         sg.Button('Stop'),
         sg.Quit()]
    ]

    # Show the Window to the user
    window = sg.Window('Button callback example').Layout(layout)

    # Event loop. Read buttons, make callbacks
    while True:
        # Read the Window
        event, value = window.Read(timeout=10)
        # Take appropriate action based on button
        if event == 'Start':
            start()
        elif event == 'Stop':
            stop()
        elif event == 'Quit' or event is None:
            window.Close()
            break

        handler.checkMessage()

