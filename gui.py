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


r = Redis()

env_config = configs.LunarLander()
policy_net = PPOWrap(env_config.features, env_config.action_map, env_config.hidden)
gui_uuid = uuid.uuid4()

gatherers = {}
gatherers_progress = {}
next_free_slot = 0

handler = MessageHandler(r, 'rollout')


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


handler.register(EpisodeMessage, episode)


def gatherer_progressbars(number, max_episodes):
    pbars = []
    for i in range(number):
        pbars.append(sg.ProgressBar(max_episodes, orientation='v', size=(20, 20), key='gatherer' + str(i)))
    return pbars


def training_progress(msg):
    window.FindElement('trainer').UpdateBar(int(msg.steps))


handler.register(TrainingProgress, training_progress)


# The callback functions
def start():
    ResetMessage(gui_uuid).send(r)
    RolloutMessage(gui_uuid, 0, policy_net, env_config).send(r)


def stop():
    StopAllMessage(gui_uuid).send(r)


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
