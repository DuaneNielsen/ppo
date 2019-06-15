import struct

import numpy as np


class NumpyCoder:
    def __init__(self, num_axes, dtype):
        self.header_fmt = '>'
        for _ in range(num_axes):
            self.header_fmt += 'I'
        self.header_size = struct.calcsize(self.header_fmt)
        self.dtype = dtype

    def encode(self, observation):
        shape = struct.pack(self.header_fmt, *observation.shape)
        b = observation.tobytes()
        return shape + b

    def decode(self, encoded):
        shape = struct.unpack(self.header_fmt, encoded[:self.header_size])
        o = np.frombuffer(encoded, dtype=self.dtype, offset=self.header_size).reshape(shape)
        return o


class StepCoder:
    def __init__(self, observation_coder):
        self.o_coder = observation_coder
        self.header_fmt = '>if?'
        self.header_len = struct.calcsize(self.header_fmt)

    def encode(self, step):
        """ encode step to bstring"""
        b_ard = struct.pack(self.header_fmt, step.action, step.reward, step.done)
        b_o = self.o_coder.encode(step.observation)
        encoded = b_ard + b_o
        return encoded

    def decode(self, encoded):
        """ decode step from bstring"""
        action, reward, done = struct.unpack(self.header_fmt, encoded[:self.header_len])
        o = self.o_coder.decode(encoded[self.header_len:])
        return Step(o, action, reward, done)


class AdvancedNumpyCoder:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.len = np.ndarray(shape, dtype=dtype).nbytes
        self.slice = None

    def set_offset(self, offset):
        """
        sets the base address offset to read from the bytestring
        returns the offset of the next field
        """
        end = offset + self.len
        self.slice = slice(offset, end)
        return end

    def encode(self, observation):
        return observation.tobytes()

    def decode(self, encoded):
        ndarray = np.frombuffer(encoded[self.slice], dtype=self.dtype).reshape(self.shape)
        return ndarray


class DiscreteActionCoder:
    def __init__(self):
        self.format = '>i'
        self.slice = None

    def set_offset(self, offset):
        """
        sets the base address offset to read from the bytestring
        returns the offset of the next field
        """
        end = offset + struct.calcsize(self.format)
        self.slice = slice(offset, end)
        return end

    def encode(self, action):
        return struct.pack(self.format, action)

    def decode(self, encoded):
        action = struct.unpack(self.format, encoded[self.slice])
        return action[0]


class RewardDoneCoder:
    def __init__(self):
        self.format = '>d?'
        self.slice = None

    def set_offset(self, offset):
        """
        sets the base address offset to read from the bytestring
        returns the offset of the next field
        """
        end = offset + struct.calcsize(self.format)
        self.slice = slice(offset, end)
        return end

    def encode(self, reward, done):
        return struct.pack(self.format, reward, done)

    def decode(self, encoded):
        reward, done = struct.unpack(self.format, encoded[self.slice])
        return reward, done


class AdvancedStepCoder:
    def __init__(self, state_shape, state_dtype, action_shape, action_dtype):
        self.reward_done_coder = RewardDoneCoder()
        end = self.reward_done_coder.set_offset(0)
        self.state_coder = AdvancedNumpyCoder(shape=state_shape, dtype=state_dtype)
        end = self.state_coder.set_offset(end)
        self.action_coder = AdvancedNumpyCoder(shape=action_shape, dtype=action_dtype)
        end = self.action_coder.set_offset(end)

    def encode(self, step):
        """
        Step should contain floats, ints, or numpy arrays
        :param step:
        :return:
        """
        encoded = self.reward_done_coder.encode(step.reward, step.done)
        encoded += self.state_coder.encode(step.observation)
        encoded += self.action_coder.encode(step.action)
        return encoded

    def decode(self, encoded):
        reward, done = self.reward_done_coder.decode(encoded)
        state = self.state_coder.decode(encoded)
        action = self.action_coder.decode(encoded)
        return Step(state, action, reward, done)


class DiscreteStepCoder:
    def __init__(self, state_shape, state_dtype):
        self.reward_done_coder = RewardDoneCoder()
        end = self.reward_done_coder.set_offset(0)
        self.state_coder = AdvancedNumpyCoder(shape=state_shape, dtype=state_dtype)
        end = self.state_coder.set_offset(end)
        self.action_coder = DiscreteActionCoder()
        end = self.action_coder.set_offset(end)

    def encode(self, step):
        """
        Step should contain floats, ints, or numpy arrays
        :param step:
        :return:
        """
        encoded = self.reward_done_coder.encode(step.reward, step.done)
        encoded += self.state_coder.encode(step.observation)
        encoded += self.action_coder.encode(step.action)
        return encoded

    def decode(self, encoded):
        reward, done = self.reward_done_coder.decode(encoded)
        state = self.state_coder.decode(encoded)
        action = self.action_coder.decode(encoded)
        return Step(state, action, reward, done)


class Step:
    def __init__(self, observation, action, reward, done):
        self.observation = observation
        self.reward = reward
        self.action = action
        self.done = done