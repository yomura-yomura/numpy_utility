import numpy as np


__all__ = ["norm"]


def norm(x, ord=None, axis=None, keepdims=False):
    """
    MaskedArray対応
    """
    if isinstance(x, np.ma.MaskedArray):
        # if x.mask.any():
        return np.ma.MaskedArray(np.linalg.norm(x.data, ord, axis, keepdims), x.mask.any(axis=axis))
    return np.linalg.norm(x, ord, axis, keepdims)
