import pytest
from messages import *
from uuid import uuid4
import configs
import torch
from models import PPOWrapModel


@pytest.fixture
def config():
    return configs.Hopper()


@pytest.fixture
def uid():
    return uuid4()


@pytest.fixture
def policy(config):
    model = config.model.get_model()
    policy = PPOWrapModel(model)
    return policy


def send_message(msg):
    encoded = JSONPickleCoder.encode(msg)
    return msg, JSONPickleCoder.decode(encoded)


def test_start_message(uid, config):

    msg, decoded_msg = send_message(StartMessage(uid, config))

    assert msg.header == StartMessage.header
    assert msg.header == decoded_msg.header
    assert msg.server_uuid == decoded_msg.server_uuid
    assert msg.config.gym_env_string ==  decoded_msg.config.gym_env_string
    assert msg.config.model_precision == decoded_msg.config.model_precision


def test_config_update_message(uid, config):

    msg, decoded_msg = send_message(ConfigUpdateMessage(uid, config))

    assert msg.header == ConfigUpdateMessage.header
    assert msg.header == decoded_msg.header
    assert msg.server_uuid == decoded_msg.server_uuid
    assert msg.config.gym_env_string == decoded_msg.config.gym_env_string


def test_train_complete_msg(config, policy):
    msg, decoded_msg = send_message(TrainCompleteMessage(uuid4(), policy, config))

    assert torch.equal(policy.new.l1_mu.weight,  decoded_msg.policy.new.l1_mu.weight)


def test_train_msg(uid, config, policy):
    msg, decoded_msg = send_message(TrainMessage(uid, policy, config))

    assert torch.equal(policy.new.l1_mu.weight,  decoded_msg.policy.new.l1_mu.weight)


def test_train_progress(uid):
    msg, decoded_msg = send_message(TrainingProgress(uid, 100000))

    assert msg.server_uuid == decoded_msg.server_uuid
    assert decoded_msg.steps == 100000


def test_simple_messages():

    clzez = [StopMessage, StopAllMessage, PingMessage, ResetMessage, StoppedMessage, ExitMessage]
    for clz in clzez:
        msg, decoded_msg = send_message(clz(uid))

        assert msg.header == clz.header
        assert msg.header == decoded_msg.header
        assert msg.server_uuid == decoded_msg.server_uuid


def test_episode_message(uid):
    msg = EpisodeMessage(uid, 'CartPole-v1_345', 2000, 3000, 53.7, 30)
    msg.monitor['entropy'] = 0.4334643554
    msg.monitor['loss'] = 0.34343e-6

    msg, decoded_msg = send_message(msg)

    assert msg.server_uuid == decoded_msg.server_uuid
    assert decoded_msg.run == 'CartPole-v1_345'
    assert decoded_msg.id == 2000
    assert decoded_msg.steps == 3000
    assert decoded_msg.total_reward == 53.7
    assert decoded_msg.num_steps_per_rollout == 30
    assert decoded_msg.monitor['loss'] == 0.34343e-6
    assert decoded_msg.monitor['entropy'] == 0.4334643554


def test_rollout_message(uid, config, policy):

    msg, decoded_msg = send_message(RolloutMessage(uuid4(), 'CartPole-v1_345', 100, policy, config, 1000))

    assert torch.equal(policy.new.l1_mu.weight,  decoded_msg.policy.new.l1_mu.weight)
    assert decoded_msg.rollout_id == 100
    assert decoded_msg.episodes == 1000
    assert decoded_msg.run == 'CartPole-v1_345'

