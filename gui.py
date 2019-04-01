import PySimpleGUI as sg
from controller import ResetMessage, StopAllMessage, RolloutMessage, MessageDecoder, EpisodeMessage, MessageHandler
from redis import Redis
import configs
from models import PPOWrap
import uuid

# This design pattern simulates button callbacks
# Note that callbacks are NOT a part of the package's interface to the
# caller intentionally.  The underlying implementation actually does use
# tkinter callbacks.  They are simply hidden from the user.

r = Redis()

env_config = configs.LunarLander()
policy_net = PPOWrap(env_config.features, env_config.action_map, env_config.hidden)
gui_uuid = uuid.uuid4()

gatherers = {}
next_free_slot = 0

handler = MessageHandler(r, 'rollout')


def episode(msg):
    global next_free_slot
    print(msg.server_uuid, msg.id)

    if msg.server_uuid not in gatherers:
        gatherers[msg.server_uuid] = 'gatherer' + str(next_free_slot)
        next_free_slot += 1

    window.FindElement(gatherers[msg.server_uuid]).UpdateBar(int(msg.id))


handler.register(EpisodeMessage, episode)


def gatherer_progressbars(number):
    pbars = []
    for i in range(number):
        pbars.append(sg.ProgressBar(10, orientation='v', size=(20, 20), key='gatherer' + str(i)))
    return pbars


# The callback functions
def start():
    ResetMessage(gui_uuid).send(r)
    RolloutMessage(gui_uuid, 0, policy_net, env_config).send(r)


def stop():
    StopAllMessage(gui_uuid).send(r)


# Layout the design of the GUI
layout = [
    [sg.Text('Please click a button', auto_size_text=True)],
    gatherer_progressbars(5),
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
