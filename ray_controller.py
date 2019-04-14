import ray
import configs
from models import PPOWrap
import os
import gym
from rollout import single_episode

@ray.remote
class ExperienceEnv(object):
    def __init__(self, env_config):
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        self.env_config = env_config
        self.policy = PPOWrap(env_config.features, env_config.action_map, env_config.hidden)
        self.env = gym.make(self.env_config.gym_env_string)

    def rollout(self, policy_weights):
        self.policy.load_state_dict(policy_weights)
        single_episode(self.env, self.env_config, self.policy)


if __name__ == "__main__":
    ray.init()

    env_config = configs.LunarLander()

    policy = PPOWrap(env_config.features, env_config.action_map, env_config.hidden)
    policy_weights = policy.state_dict()

    gatherers = [ExperienceEnv.remote(env_config) for _ in range(env_config.experience_threads)]

    #rollout = None

    for i in range(env_config.experience_threads):
        gatherers[i].rollout.remote(policy_weights)










