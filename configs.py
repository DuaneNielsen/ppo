import datetime

import cv2
from torchvision.transforms.functional import to_tensor
import gym, roboschool
import data
import data.transforms
from data.prepro import DefaultPrePro, NoPrePro
from algos import PurePPOClip, OneStepTD
from data.transforms import DefaultTransform, ContinousActionTransform
from data.coders import DiscreteStepCoder, AdvancedStepCoder
import numpy as np
import torch
from models import SmartQTable, RandomDiscretePolicy, MultiPolicyNetContinuous, RandomContinuousPolicy, ValuePolicy, \
    EpsilonGreedyDiscreteDist
from jsonpickle.pickler import Pickler
import algos

class ConfigItem:
    pass


class DataConfigItem(ConfigItem):
    def __init__(self, cls, *args, **kwargs):
        self.name = cls.__name__
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    def construct(self):
        return self.cls(*self.args, **self.kwargs)


class NoOptimizer(Exception):
    pass


class OptimizerConfig(ConfigItem):
    def __init__(self, clazz, **kwargs):
        self.name = clazz.__name__
        self.clazz = clazz
        self.kwargs = kwargs

    def construct(self, parameters):
        return self.clazz(params=parameters, **self.kwargs)


adam = OptimizerConfig(torch.optim.Adam, lr=1e-3)
sgd = OptimizerConfig(torch.optim.SGD, lr=0.1)


class AlgoConfig(ConfigItem):
    def __init__(self, algo_class, **kwargs):
        self.clazz = algo_class
        self.name = algo_class.__name__
        self.kwargs = kwargs

    def construct(self):
        return self.clazz(**self.kwargs)


class PurePPOClipConfig(AlgoConfig):
    def __init__(self, optimizer=sgd, discount_factor=0.99, ppo_steps_per_batch=10):
        super().__init__(PurePPOClip)
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        self.ppo_steps_per_batch = ppo_steps_per_batch


class OneStepTDConfig(AlgoConfig):
    def __init__(self, optimizer=adam, epsilon=0.05, discount_factor=0.99, min_change=2e-4, detections=5,
                 detection_window=8):
        super().__init__(OneStepTD)
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.min_change = min_change
        self.detections = detections
        self.detection_window = detection_window
        self.logging_freq = 1000


class BoostrapValueConfig(AlgoConfig):
    def __init__(self, optimizer=adam, discount_factor=0.99, min_change=2e-4, detections=5,
                 detection_window=8):
        super().__init__(algos.BootstrapValue)
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        self.min_change = min_change
        self.detections = detections
        self.detection_window = detection_window
        self.logging_freq = 1000


class PPOA2CConfig(AlgoConfig):
    def __init__(self, optimizer=adam, discount_factor=0.99, min_change=2e-4, detections=5,
                 detection_window=8, ppo_steps_per_batch=10, logging_freq=100):
        super().__init__(algos.PPOAC2)
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        self.min_change = min_change
        self.detections = detections
        self.detection_window = detection_window
        self.logging_freq = logging_freq
        self.ppo_steps_per_batch = ppo_steps_per_batch


class ModelConfig(ConfigItem):
    def __init__(self, clazz, *args, **kwargs):
        self.clazz = clazz
        self.name = clazz.__name__
        self.args = args
        self.kwargs = kwargs

    def get_model(self):
        return self.clazz(*self.args, **self.kwargs)

    def construct(self):
        return self.clazz(*self.args, **self.kwargs)


class ValuePolicyConfig(ModelConfig):
    def __init__(self, qfunc_config, distrib, epsilon):
        super().__init__(ValuePolicyConfig)
        self.qfunc_config = qfunc_config
        self.distrib = distrib
        self.epsilon = epsilon

    def construct(self):
        qfunc = self.qfunc_config.construct()
        return ValuePolicy(qfunc, self.distrib, epsilon=self.epsilon)


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


class EnvConfig(ConfigItem):
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


class DataConfig(ConfigItem):
    def __init__(self, coder, prepro, transform, action_transform):
        self.coder = coder
        self.prepro = prepro
        self.transform = transform
        self.action_transform = action_transform
        self._precision = "torch.float32"

    @property
    def precision(self):
        return eval(self._precision)

    def transforms(self):
        return self.transform.construct(), self.action_transform.construct()


class GatherConfig(ConfigItem):
    def __init__(self):
        self.episode_batch_size = 1  # the number of steps to buffer in the gatherer before updating
        self.episodes_per_gatherer = 1  # number of episodes to gather before waiting for co-ordinator
        self.num_steps_per_rollout = 2000
        self.policy_reservoir_depth = 10
        self.policy_top_depth = 10

        self.max_rollout_len = 3000  # terminate episodes that go longer than this


class BaseConfig(ConfigItem):
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

        #timeout for system retry
        self.timeout = 400

        # gpu diagnostics
        self.gpu_profile = False
        self.gpu_profile_fn = f'{datetime.datetime.now():%d-%b-%y-%H-%M-%S}-gpu_mem_prof.txt'
        self.lineno = None
        self.func_name = None
        self.filename = None
        self.module_name = None

    @staticmethod
    def recurse_view(obj):
        view = {}
        if isinstance(obj, dict):
            d = obj
        else:
            d = obj.__dict__

        for name, value in d.items():
            if issubclass(type(value), ConfigItem):
                view[name] = BaseConfig.recurse_view(value)
            elif name is 'kwargs':
                BaseConfig.recurse_view(value)
            elif isinstance(value, str) or isinstance(value, float) or isinstance(value, int) \
                    or isinstance(value, tuple) or isinstance(value, np.dtype) or isinstance(value, list):
                view[name] = value
            elif isinstance(value, type):
                view[name] = value.__name__
            else:
                view[name] = type(value).__name__

        return view

    def view(self):
        return Pickler().flatten(self)


class Discrete(BaseConfig):
    def __init__(self, env_string, wrappers=None):
        env_config = make_env_config_for(env_string, wrappers)
        critic_config = ModelConfig(SmartQTable, features=env_config.state_space_shape[0], actions=env_config.actions,
                                    resnet_layers=1)
        random_policy_config = ModelConfig(RandomDiscretePolicy, env_config.actions)
        algo_config = OneStepTDConfig()
        actor_config = ValuePolicyConfig(critic_config, EpsilonGreedyDiscreteDist, algo_config.epsilon)
        data_config = DataConfig(
            coder=DataConfigItem(DiscreteStepCoder,
                                 state_shape=env_config.state_space_shape,
                                 state_dtype=env_config.state_space_dtype),
            prepro=DataConfigItem(DefaultPrePro),
            transform=DataConfigItem(data.transforms.DefaultTransform),
            action_transform=DataConfigItem(data.transforms.OneHotDiscreteActionTransform, env_config.action_map)
        )
        gatherer_config = GatherConfig()
        super().__init__(env_config, algo_config, random_policy_config, actor_config, critic_config, data_config,
                         gatherer_config)


default = Discrete('LunarLander-v2')


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
            coder=DataConfigItem(AdvancedStepCoder,
                                 state_shape=env_config.state_space_shape,
                                 state_dtype=env_config.state_space_dtype,
                                 action_shape=env_config.action_space_shape,
                                 action_dtype=env_config.action_space_dtype),
            prepro=DataConfigItem(DefaultPrePro),
            transform=DataConfigItem(DefaultTransform),
            action_transform=DataConfigItem(ContinousActionTransform)
        )
        gatherer_config = GatherConfig()
        super().__init__(env_config, algo_config, random_policy_config, actor_config, critic_config, data_config,
                         gatherer_config)


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
