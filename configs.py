import datetime

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor
import gym
import data
from data import SingleProcessDataSet, StepCoder, NumpyCoder, AdvancedStepCoder
import logging
import models


def load_model(config, parameters):
    model = config.model.get_model()
    return model.load_state_dict(parameters)


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


class InfinityException(Exception):
    pass


class ContinousActionTransform:
    def __call__(self, action):
        action = action.squeeze()
        if torch.isnan(action).any().item() == 1:
            raise InfinityException
        return action.numpy()


class ModelConfig:
    def __init__(self, name):
        self.name = name

    def get_model(self):
        return None


class AlgoConfig:
    def __init__(self, name):
        self.name=name


class EnvConfig:
    def __init__(self, name):
        self.name=name


class MultiPolicyNet(ModelConfig):
    def __init__(self, features, hidden_size, actions):
        super().__init__('MultiPolicyNet')
        self.features = features
        self.hidden_size = hidden_size
        self.actions = actions

    def get_model(self):
        return models.MultiPolicyNet(self.features, self.actions, self.hidden_size)


class MultiPolicyNetContinuous(ModelConfig):
    def __init__(self, feature_size, hidden_size, action_size):
        super().__init__('MultiPolicyNet')
        self.features_size = feature_size
        self.hidden_size = hidden_size
        self.action_size = action_size

    def get_model(self):
        return models.MultiPolicyNetContinuous(self.features_size, self.action_size, self.hidden_size)


class MultiPolicyNetContinuousV2(ModelConfig):
    def __init__(self, feature_size, hidden_size, action_size):
        super().__init__('MultiPolicyNetContinuousV2')
        self.features_size = feature_size
        self.hidden_size = hidden_size
        self.action_size = action_size

    def get_model(self):
        return models.MultiPolicyNetContinuousV2(self.features_size, self.action_size, self.hidden_size)


class GymEnvConfig(EnvConfig):
    def __init__(self, name):
        super().__init__(name)


class BaseConfig:
    def __init__(self,
                 gym_env_string,
                 discount_factor=0.99,
                 prepro=DefaultPrePro(),
                 transform=DefaultTransform(),
                 ):

        # environment config
        self.gym_env_string = gym_env_string

        self.training_algo = 'ppo'
        self.model_string = 'MultiPolicyNet'
        self.step_coder = None
        self.discount_factor = discount_factor
        self.max_rollout_len = 3000  #terminate episodes that go longer than this
        self.prepro = prepro
        self.transform = transform

        self.continuous = False

        # gathering
        self.episode_batch_size = 1  # the number of steps to buffer in the gatherer before updating
        self.episodes_per_gatherer = 1 # number of episodes to gather before waiting for co-ordinator
        self.num_steps_per_rollout = 2000
        self.policy_reservoir_depth = 10
        self.policy_top_depth = 10
        self.run_id = ''
        self.timeout = 40

        # training
        self.max_minibatch_size = 10000
        self.optimizer = 'SGD'
        self.lr = 0.05
        self.ppo_steps_per_batch = 10
        self.entropy_bonus = 0.0

        self.gpu_profile = False
        self.gpu_profile_fn = f'{datetime.datetime.now():%d-%b-%y-%H-%M-%S}-gpu_mem_prof.txt'
        self.lineno = None
        self.func_name = None
        self.filename = None
        self.module_name = None


class DiscreteConfig(BaseConfig):
    def __init__(self,
                 gym_env_string,
                 features,
                 features_dtype,
                 action_map,
                 default_action=0,
                 discount_factor=0.99,
                 prepro=DefaultPrePro(),
                 transform=DefaultTransform(),
                 ):
        super().__init__(gym_env_string, discount_factor, prepro, transform)
        self.features = features
        self.action_map = action_map
        self.actions = len(action_map)
        self.default_action = torch.tensor([default_action])
        self.model = MultiPolicyNet(features, features, self.actions)
        self.action_transform = models.DiscreteActionTransform(action_map)
        self.step_coder = AdvancedStepCoder(state_shape=features, state_dtype=features_dtype,
                                            action_shape=(1,), action_dtype=np.int64)


class ContinuousConfig(BaseConfig):
    def __init__(self,
                 gym_env_string,
                 state_space_features,
                 state_space_dtype,
                 action_space_features,
                 action_space_dtype,
                 discount_factor=0.99,
                 prepro=DefaultPrePro(),
                 transform=DefaultTransform(),
                 ):
        super().__init__(gym_env_string, discount_factor, prepro, transform)
        self.state_space_features = state_space_features
        self.state_space_dtype = state_space_dtype
        self.action_space_features = action_space_features
        self.action_space_dtype = action_space_dtype
        self.continuous = True
        self.action_transform = ContinousActionTransform()
        self.default_action = torch.zeros(action_space_features)
        self.step_coder = AdvancedStepCoder(state_shape=self.state_space_features, state_dtype=self.state_space_dtype,
                                            action_shape=self.action_space_features, action_dtype=self.action_space_dtype)


class HalfCheetah(ContinuousConfig):
    def __init__(self):
        super().__init__(
            gym_env_string='RoboschoolHalfCheetah-v1',
            state_space_features=26,
            state_space_dtype=np.float32,
            action_space_features=6,
            action_space_dtype=np.float32
        )
        self.model = MultiPolicyNetContinuousV2(
            feature_size=self.state_space_features,
            hidden_size=26,
            action_size=self.action_space_features
        )


class Hopper(ContinuousConfig):
    def __init__(self):
        super().__init__(
            gym_env_string='RoboschoolHopper-v1',
            state_space_features=15,
            state_space_dtype=np.float32,
            action_space_features=3,
            action_space_dtype=np.float32
        )
        self.model = MultiPolicyNetContinuousV2(
            feature_size=self.state_space_features,
            hidden_size=self.state_space_features,
            action_size=self.action_space_features
        )


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


class GymDiscreteConfig(DiscreteConfig):
    def __init__(self, gym_string):
        env = gym.make(gym_string)
        dtype = env.reset().dtype
        super().__init__(
            features=env.observation_space.shape[0],
            features_dtype=dtype,
            gym_env_string=gym_string,
            action_map=[n for n in range(env.action_space.n)]
        )
        self.hidden = 8
        self.adversarial = False
        self.players = 1


class LunarLander(GymDiscreteConfig):
    def __init__(self):
        super().__init__('LunarLander-v2')


class CartPole(GymDiscreteConfig):
    def __init__(self):
        super().__init__('CartPole-v0')


class Acrobot(GymDiscreteConfig):
    def __init__(self):
        super().__init__('Acrobot-v1')


class MountainCar(DiscreteConfig):
    def __init__(self):
        gym_string = 'MountainCar-v0'
        env = gym.make(gym_string)
        dtype = env.reset().dtype
        super().__init__(
            features=env.observation_space.shape[0],
            features_dtype=dtype,
            gym_env_string=gym_string,
            action_map=[n for n in range(env.action_space.n)]
        )
        self.hidden = 8
        self.adversarial = False
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


default = CartPole()
