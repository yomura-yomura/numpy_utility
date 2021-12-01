import numpy as np


__all__ = ["trunc", "add", "subtract"]


def trunc(x, decimals=0):
    factor = np.power(10, decimals)
    return np.trunc(np.multiply(x, factor)) / factor


def add(x1, x2, *args, **kwargs):
    if not isinstance(x1, np.ma.MaskedArray):
        x1 = np.asarray(x1)

    if x1.dtype.names is None:
        new_x = np.add(x1, x2, *args, **kwargs)
    else:
        if not isinstance(x2, np.ma.MaskedArray):
            x2 = np.asarray(x2)
        assert x1.dtype == x2.dtype
        if isinstance(x1, np.ma.MaskedArray) or isinstance(x2, np.ma.MaskedArray):
            new_x = np.ma.empty_like(x1)
        else:
            new_x = np.empty_like(x1)
        for name in x1.dtype.names:
            new_x[name] = add(x1[name], x2[name], *args, **kwargs)
    return new_x


def subtract(x1, x2, *args, **kwargs):
    if not isinstance(x1, np.ma.MaskedArray):
        x1 = np.asarray(x1)

    if x1.dtype.names is None:
        new_x = np.subtract(x1, x2, *args, **kwargs)
    else:
        if not isinstance(x2, np.ma.MaskedArray):
            x2 = np.asarray(x2)
        assert x1.dtype == x2.dtype
        if isinstance(x1, np.ma.MaskedArray) or isinstance(x2, np.ma.MaskedArray):
            new_x = np.ma.empty_like(x1)
        else:
            new_x = np.empty_like(x1)
        for name in x1.dtype.names:
            new_x[name] = subtract(x1[name], x2[name], *args, **kwargs)
    return new_x

