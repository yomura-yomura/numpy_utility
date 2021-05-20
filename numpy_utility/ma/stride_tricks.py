import numpy as np


__all__ = ["broadcast_arrays"]


def broadcast_arrays(*args):
    arrays = np.broadcast_arrays(*args, subok=True)
    for a, oa in zip(arrays, args):
        if isinstance(a, np.ma.MaskedArray):
            a.mask = np.broadcast_to(oa.mask, a.shape)
    return arrays
