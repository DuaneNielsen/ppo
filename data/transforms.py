import torch


class DefaultTransform:
    def __call__(self, observation, dtype=torch.float32, insert_batch=False):
        """
        env -> model tensor
        :param observation: the raw observation
        :param insert_batch: add a batch dimension to the front
        :return: tensor in shape (batch, dims)
        """
        if insert_batch:
            return torch.from_numpy(observation).to(dtype=dtype).unsqueeze(0)
        else:
            return torch.from_numpy(observation).to(dtype=dtype)


class InfinityException(Exception):
    pass


class ContinousActionTransform:
    def __call__(self, action):
        action = action.squeeze()
        if torch.isnan(action).any().item() == 1:
            raise InfinityException
        return action.numpy()

    def invert(self, action, dtype):
        return torch.from_numpy(action).to(dtype=dtype)


class DiscreteActionTransform:
    def __init__(self, action_map):
        """
        Takes a single action from model and converts it to an integer mapped
        to the environments action space
        :param action_map:
        """
        self.action_map = torch.tensor(action_map)

        self.reverse = [0] * (max(action_map) + 1)

        for i, item in enumerate(action_map):
            self.reverse[item] = i

    def __call__(self, index):
        """
        model -> environment
        :param index:
        :return:
        """
        action_map = self.action_map.expand(index.size(0), -1)
        action = action_map.take(index)
        return action.squeeze().item()

    def invert(self, action, dtype=None):
        """
        env -> model
        converts the action on the environment action space back to the model action space
        :param action:
        :return:
        """
        return torch.tensor([self.reverse[action]])


class OneHotDiscreteActionTransform:
    def __init__(self, action_map):
        self.action_map = torch.tensor(action_map)

        self.reverse = [0] * (max(action_map) + 1)

        for i, item in enumerate(action_map):
            self.reverse[item] = i

    def __call__(self, action):
        """
        model -> environment
        :return:
        """
        return self.action_map[torch.argmax(action.squeeze())].item()

    def invert(self, action, dtype):
        """
        converts one action from env -> model
        :param action:
        :param dtype:
        :return:
        """
        one_hot = torch.zeros(self.action_map.size(0), dtype=dtype)
        index = self.reverse[action]
        one_hot[index] = 1.0
        return one_hot