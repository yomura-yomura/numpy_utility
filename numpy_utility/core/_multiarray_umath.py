import numpy as np


__all__ = ["trunc"]


def trunc(x, decimals=0):
    factor = np.power(10, decimals)
    return np.trunc(np.multiply(x, factor)) / factor
