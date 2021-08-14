import numpy as np


__all__ = ["norm", "dot"]


def norm(x, ord=None, axis=None, keepdims=False):
    """
    MaskedArray対応
    """
    if isinstance(x, np.ma.MaskedArray):
        # if x.mask.any():
        return np.ma.MaskedArray(np.linalg.norm(x.data, ord, axis, keepdims), x.mask.any(axis=axis))
    return np.linalg.norm(x, ord, axis, keepdims)


def dot(a, b, axis=-1, keepdims=False):
    if keepdims:
        return np.expand_dims(dot(a, b, axis, False), axis)
    return np.sum(a * b, axis=axis)
