"""Normalization methods used by the main training workflow."""

import numpy as np
import torch


class Scaler:
    """Interface shared by all normalization methods."""

    def transform(self, data):
        raise NotImplementedError

    def inverse_transform(self, data):
        raise NotImplementedError


class NoneScaler(Scaler):
    """Leave the data unchanged."""

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class NormalScaler(Scaler):
    """Divide values by the training maximum."""

    def __init__(self, maxx):
        self.max = maxx

    def transform(self, data):
        return data / self.max

    def inverse_transform(self, data):
        return data * self.max


class StandardScaler(Scaler):
    """Apply z-score normalization."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class MinMax01Scaler(Scaler):
    """Scale values to the interval [0, 1]."""

    def __init__(self, minn, maxx):
        self.min = minn
        self.max = maxx

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        return data * (self.max - self.min) + self.min


class MinMax11Scaler(Scaler):
    """Scale values to the interval [-1, 1]."""

    def __init__(self, minn, maxx):
        self.min = minn
        self.max = maxx

    def transform(self, data):
        return ((data - self.min) / (self.max - self.min)) * 2. - 1.

    def inverse_transform(self, data):
        return ((data + 1.) / 2.) * (self.max - self.min) + self.min


class LogScaler(Scaler):
    """Apply ``log(data + eps)`` and its inverse."""

    def __init__(self, eps=0.999):
        self.eps = eps

    @staticmethod
    def _to_numpy(data):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return data

    def transform(self, data):
        return np.log(self._to_numpy(data) + self.eps)

    def inverse_transform(self, data):
        return np.exp(self._to_numpy(data)) - self.eps
