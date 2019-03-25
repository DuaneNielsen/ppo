import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

import data
from data import SingleProcessDataSet, StepCoder, NumpyCoder


class DefaultPrePro:
    def __call__(self, observation_t1, observation_t0):
        return observation_t1 - observation_t0


class DefaultTransform:
    def __call__(self, observation, insert_batch=False):
        """
        :param observation: the raw observation
        :param insert_batch: add a batch dimension to the front
        :return: tensor in shape (batch, dims)
        """
        if insert_batch:
            return torch.from_numpy(observation).float().unsqueeze(0)
        else:
            return torch.from_numpy(observation).float()


class BaseConfig:
    def __init__(self,
                 gym_env_string,
                 step_coder,
                 discount_factor=0.99,
                 max_rollout_len=3000,
                 prepro=DefaultPrePro(),
                 transform=DefaultTransform(),
                 ):
        self.gym_env_string = gym_env_string
        self.step_coder = step_coder
        self.discount_factor = discount_factor
        self.max_rollout_len = max_rollout_len
        self.prepro = prepro
        self.transform = transform


class DiscreteConfig(BaseConfig):
    def __init__(self,
                 gym_env_string,
                 step_coder,
                 action_map,
                 default_action=0,
                 discount_factor=0.99,
                 max_rollout_len=3000,
                 prepro=DefaultPrePro(),
                 transform=DefaultTransform(),
                 ):
        super().__init__(gym_env_string, step_coder, discount_factor, max_rollout_len, prepro, transform)
        self.action_map = action_map
        self.default_action = default_action


class Pong:
    def __init__(self):
        self.gym_env_string = 'Pong-v0'
        self.downsample_image_size = (100, 80)
        self.features = self.downsample_image_size[0] * self.downsample_image_size[1]
        self.hidden = 200
        self.action_map = [0, 1, 2, 3]
        self.default_action = 0
        self.discount_factor = 0.99
        self.max_rollout_len = 3000

    def prepro(self, observation):
        greyscale = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        greyscale = cv2.resize(greyscale, self.downsample_image_size, cv2.INTER_LINEAR)
        return greyscale

    def transform(self, observation, insert_batch=False):
        if insert_batch:
            return to_tensor(np.expand_dims(observation, axis=2)).view(1, self.features)
        else:
            return to_tensor(np.expand_dims(observation, axis=2)).view(self.features)


class LunarLander(DiscreteConfig):
    def __init__(self):
        super().__init__(
            gym_env_string='LunarLander-v2',
            step_coder=StepCoder(observation_coder=NumpyCoder(1, np.float32)),
            action_map=[0, 1, 2, 3]
        )
        self.features = 8
        self.hidden = 8
        self.adversarial = False
        self.default_save = ['lunar_lander/solved.wgt']
        self.players = 1


class PongAdversarial:
    def __init__(self):
        self.gym_env_string = 'PymunkPong-v0'
        self.downsample_image_size = (100, 80)
        self.features = 100 * 80
        self.hidden = 200
        self.action_map = [0, 1, 2]
        self.default_action = 2
        self.discount_factor = 0.99
        self.max_rollout_len = 1000
        self.players = 2
        self.default_save = ['saved/adv_pong.wgt']

    def construct_dataset(self):
        return data.BufferedRolloutDataset(self.discount_factor, transform=self.transform)

    def prepro(self, t1, t0):
        def reduce(observation):
            greyscale = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
            greyscale = cv2.resize(greyscale, self.downsample_image_size, cv2.INTER_LINEAR)
            return greyscale

        t1 = reduce(t1)
        t0 = reduce(t0)
        return t1 - t0

    def transform(self, observation, insert_batch=False):
        observation_t = to_tensor(np.expand_dims(observation, axis=2)).view(self.features)
        if insert_batch:
            observation_t = observation_t.unsqueeze(0)
        return observation_t


class AlphaDroneRacer:
    def __init__(self):
        self.gym_env_string = 'AlphaRacer2D-v0'
        self.features = 14
        self.hidden = 14
        self.action_map = [0, 1, 2, 3]
        self.default_action = 0
        self.discount_factor = 0.99
        self.max_rollout_len = 900
        self.adversarial = False
        self.default_save = ['saved/alpha_oscilating.wgt']
        self.players = 1

    def construct_dataset(self):
        return SingleProcessDataSet(self)

    def prepro(self, observation_t1, observation_t0):
        return np.concatenate((observation_t1, observation_t0))

    def transform(self, observation, insert_batch=False):
        """
        :param observation: the raw observation
        :param insert_batch: add a batch dimension to the front
        :return: tensor in shape (batch, dims)
        """
        if insert_batch:
            return torch.from_numpy(observation).float().unsqueeze(0)
        else:
            return torch.from_numpy(observation).float()


class Bouncer:
    def __init__(self):
        self.gym_env_string = 'Bouncer-v0'
        self.features = 8
        self.hidden = 8
        self.action_map = [0, 1, 2, 3, 4]
        self.default_action = 0
        self.discount_factor = 0.99
        self.max_rollout_len = 900
        self.adversarial = False
        self.default_save = ['saved/bouncer.wgt']
        self.players = 1

    def construct_dataset(self):
        return SingleProcessDataSet(self)

    def prepro(self, observation_t1, observation_t0):
        return np.concatenate((observation_t1, observation_t0))

    def transform(self, observation, insert_batch=False):
        """
        :param observation: the raw observation
        :param insert_batch: add a batch dimension to the front
        :return: tensor in shape (batch, dims)
        """
        if insert_batch:
            return torch.from_numpy(observation).float().unsqueeze(0)
        else:
            return torch.from_numpy(observation).float()
