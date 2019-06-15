class DefaultPrePro:
    def __call__(self, observation_t1, observation_t0):
        return observation_t1 - observation_t0


class NoPrePro:
    def __call__(self, observation_t1, observation_t0):
        return observation_t1