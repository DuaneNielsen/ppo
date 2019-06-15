import datetime

import cv2
from torchvision.transforms.functional import to_tensor
import gym, roboschool
import data
import data.transforms
from data.prepro import DefaultPrePro, NoPrePro
from algos import OneStepTDConfig, PurePPOClipConfig, OptimizerConfig
from data.transforms import DefaultTransform, ContinousActionTransform
from data.coders import DiscreteStepCoder, AdvancedStepCoder
import numpy as np
import torch
from models import SmartQTable, RandomDiscretePolicy, MultiPolicyNetContinuous, RandomContinuousPolicy


class ModelConfig:
    def __init__(self, clazz, *args, **kwargs):
        self.clazz = clazz
        self.name = clazz.__name__
        self.args = args
        self.kwargs = kwargs

    def get_model(self):
        return self.clazz(*self.args, **self.kwargs)

    def construct(self):
        return self.clazz(*self.args, **self.kwargs)


class NoModel(ModelConfig):
    def __init__(self):
        super().__init__(str)

    def get_model(self):
        return None

    def construct(self):
        return None


class UnknownSpaceTypeException(Exception):
    pass


# todo this might also be able to return a dataconfig
# todo account for preprocessing
def make_env_config_for(name, wrappers=None):
    env = gym.make(name)
    if wrappers is not None:
        for wrapper in wrappers:
            env = wrapper(env)

    if isinstance(env.action_space, gym.spaces.Discrete):
        env_config = GymDiscreteConfig(name, env)
    elif isinstance(env.action_space, gym.spaces.Box):
        env_config = GymContinuousConfig(name, env)
    else:
        raise UnknownSpaceTypeException()

    env_config.wrappers = wrappers if wrappers is not None else []

    return env_config


def make_connector_config_for(config):
    pass


class EnvConfig:
    def __init__(self, name, env):
        self.name = name
        obs = env.reset()
        self.state_space_shape = obs.shape
        self.state_space_dtype = obs.dtype
        self.wrappers = []
        self.default_action = None

    def construct(self):
        env = gym.make(self.name)
        for wrapper in self.wrappers:
            env = wrapper(env)
        return env


class GymDiscreteConfig(EnvConfig):
    def __init__(self, name, env):
        super().__init__(name, env)
        self.action_map = [n for n in range(env.action_space.n)]
        self.actions = env.action_space.n
        self.default_action = 0


class DataConfig:
    def __init__(self, coder, prepro, transform, action_transform):
        self.coder = coder
        self.prepro = prepro
        self.transform = transform
        self.action_transform = action_transform
        self._precision = "torch.float32"

    @property
    def precision(self):
        return eval(self._precision)


class GatherConfig:
    def __init__(self):
        self.episode_batch_size = 1  # the number of steps to buffer in the gatherer before updating
        self.episodes_per_gatherer = 1  # number of episodes to gather before waiting for co-ordinator
        self.num_steps_per_rollout = 2000
        self.policy_reservoir_depth = 10
        self.policy_top_depth = 10
        self.timeout = 40
        self.max_rollout_len = 3000  # terminate episodes that go longer than this


class BaseConfig:
    def __init__(self,
                 env_config,
                 algo_config,
                 random_policy_config,
                 actor_config,
                 critic_config,
                 data_config,
                 gatherer_config
                 ):
        # run id
        self.run_id = ''

        # environment config
        self.env = env_config

        # model used for critic
        self.critic = critic_config

        # model used for actor
        self.actor = actor_config

        # training algorithm config
        self.algo = algo_config

        # random policy used to gather intial run
        self.random_policy = random_policy_config

        # data pipeline config
        self.data = data_config

        # configuration of data run
        self.gatherer = gatherer_config

        # gpu diagnostics
        self.gpu_profile = False
        self.gpu_profile_fn = f'{datetime.datetime.now():%d-%b-%y-%H-%M-%S}-gpu_mem_prof.txt'
        self.lineno = None
        self.func_name = None
        self.filename = None
        self.module_name = None


class Discrete(BaseConfig):
    def __init__(self, env_string, wrappers=None):
        env_config = make_env_config_for(env_string, wrappers)
        actor_config = NoModel()
        critic_config = ModelConfig(SmartQTable, features=env_config.state_space_shape[0], actions=env_config.actions,
                                    resnet_layers=1)
        random_policy_config = ModelConfig(RandomDiscretePolicy, env_config.actions)
        algo_config = OneStepTDConfig()
        data_config = DataConfig(
            coder=DiscreteStepCoder(state_shape=env_config.state_space_shape, state_dtype=env_config.state_space_dtype),
            prepro=DefaultPrePro(),
            transform=data.transforms.DefaultTransform(),
            action_transform=data.transforms.OneHotDiscreteActionTransform(env_config.action_map)
        )
        gatherer_config = GatherConfig()
        super().__init__(env_config, algo_config, random_policy_config, actor_config, critic_config, data_config, gatherer_config)


class GymContinuousConfig(EnvConfig):
    def __init__(self, name, env):
        super().__init__(name, env)
        self.action_space_shape = env.action_space.shape
        self.action_space_dtype = env.action_space.dtype
        self.default_action = np.zeros(env.action_space.shape)


class Continuous(BaseConfig):
    def __init__(self, env_string, wrappers=None):
        env_config = make_env_config_for(env_string, wrappers)
        actor_config = ModelConfig(MultiPolicyNetContinuous, env_config.state_space_shape[0],
                                    env_config.action_space_shape[0], env_config.state_space_shape[0])
        critic_config = NoModel()
        random_policy_config = ModelConfig(RandomContinuousPolicy, env_config.action_space_shape)
        algo_config = PurePPOClipConfig()
        data_config = DataConfig(
            coder=AdvancedStepCoder(state_shape=env_config.state_space_shape, state_dtype=env_config.state_space_dtype,
                                    action_shape=env_config.action_space_shape, action_dtype=env_config.action_space_dtype),
            prepro=DefaultPrePro(),
            transform=DefaultTransform(),
            action_transform=ContinousActionTransform()
        )
        gatherer_config = GatherConfig()
        super().__init__(env_config, algo_config, random_policy_config, actor_config, critic_config, data_config, gatherer_config)




#
# class HalfCheetah(ContinuousConfig):
#     def __init__(self):
#         super().__init__(
#             gym_env_string='RoboschoolHalfCheetah-v1',
#             state_space_features=26,
#             state_space_dtype=np.float32,
#             action_space_features=6,
#             action_space_dtype=np.float32
#         )
#         self.model = MultiPolicyNetContinuousV2(
#             feature_size=self.state_space_features,
#             hidden_size=26,
#             action_size=self.action_space_features
#         )
#
#
# class Hopper(ContinuousConfig):
#     def __init__(self):
#         super().__init__(
#             gym_env_string='RoboschoolHopper-v1',
#             state_space_features=15,
#             state_space_dtype=np.float32,
#             action_space_features=3,
#             action_space_dtype=np.float32
#         )
#         self.model = MultiPolicyNetContinuousV2(
#             feature_size=self.state_space_features,
#             hidden_size=self.state_space_features,
#             action_size=self.action_space_features
#         )


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

# class CartPole(GymDiscreteConfig):
#     def __init__(self):
#         super().__init__('CartPole-v0')
#
#
# class Acrobot(GymDiscreteConfig):
#     def __init__(self):
#         super().__init__('Acrobot-v1')


# class MountainCar(DiscreteConfig):
#     def __init__(self):
#         gym_string = 'MountainCar-v0'
#         env = gym.make(gym_string)
#         dtype = env.reset().dtype
#         super().__init__(
#             features=env.observation_space.shape[0],
#             features_dtype=dtype,
#             gym_env_string=gym_string,
#             action_map=[n for n in range(env.action_space.n)]
#         )
#         self.hidden = 8
#         self.adversarial = False
#         self.players = 1


# class MountainCarValue(DiscreteConfig):
#     def __init__(self):
#         gym_string = 'MountainCar-v0'
#         env = gym.make(gym_string)
#         dtype = env.reset().dtype
#         super().__init__(
#             features=env.observation_space.shape[0],
#             features_dtype=dtype,
#             gym_env_string=gym_string,
#             action_map=[n for n in range(env.action_space.n)]
#         )
#         self.hidden = 8
#         self.adversarial = False
#         self.players = 1
#         self.action_transform = data.transforms.OneHotDiscreteActionTransform(self.action_map)


# class FrozenLakeValue(DiscreteConfig):
#     def __init__(self):
#         gym_string = 'FrozenLake-v0'
#         env = gym.make(gym_string)
#         env = OneHotObsWrapper(env)
#         dtype = env.reset().dtype
#         super().__init__(
#             features=env.observation_space.shape[0],
#             features_dtype=dtype,
#             gym_env_string=gym_string,
#             action_map=[n for n in range(env.action_space.n)],
#             prepro=NoPrePro()
#         )
#         self.hidden = 8
#         self.adversarial = False
#         self.players = 1
#         self.action_transform = data.transforms.OneHotDiscreteActionTransform(self.action_map)
#         self.wrappers = [OneHotObsWrapper]
#

# class PongAdversarial:
#     def __init__(self):
#         self.gym_env_string = 'PymunkPong-v0'
#         self.downsample_image_size = (100, 80)
#         self.features = 100 * 80
#         self.hidden = 200
#         self.action_map = [0, 1, 2]
#         self.default_action = 2
#         self.discount_factor = 0.99
#         self.max_rollout_len = 1000
#         self.players = 2
#         self.default_save = ['saved/adv_pong.wgt']
#
#     def construct_dataset(self):
#         return data.BufferedRolloutDataset(self.discount_factor, transform=self.transform)
#
#     def prepro(self, t1, t0):
#         def reduce(observation):
#             greyscale = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
#             greyscale = cv2.resize(greyscale, self.downsample_image_size, cv2.INTER_LINEAR)
#             return greyscale
#
#         t1 = reduce(t1)
#         t0 = reduce(t0)
#         return t1 - t0
#
#     def transform(self, observation, insert_batch=False):
#         observation_t = to_tensor(np.expand_dims(observation, axis=2)).view(self.features)
#         if insert_batch:
#             observation_t = observation_t.unsqueeze(0)
#         return observation_t


# class AlphaDroneRacer:
#     def __init__(self):
#         self.gym_env_string = 'AlphaRacer2D-v0'
#         self.features = 14
#         self.hidden = 14
#         self.action_map = [0, 1, 2, 3]
#         self.default_action = 0
#         self.discount_factor = 0.99
#         self.max_rollout_len = 900
#         self.adversarial = False
#         self.default_save = ['saved/alpha_oscilating.wgt']
#         self.players = 1
#
#     def prepro(self, observation_t1, observation_t0):
#         return np.concatenate((observation_t1, observation_t0))
#
#     def transform(self, observation, insert_batch=False):
#         """
#         :param observation: the raw observation
#         :param insert_batch: add a batch dimension to the front
#         :return: tensor in shape (batch, dims)
#         """
#         if insert_batch:
#             return torch.from_numpy(observation).float().unsqueeze(0)
#         else:
#             return torch.from_numpy(observation).float()

# default = CartPole()
