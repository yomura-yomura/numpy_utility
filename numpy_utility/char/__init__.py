import numpy as np
import functools
import more_itertools
import itertools
from ..core import is_array

__all__ = ["join"]


def is_string(a: np.ndarray):
    """
    Binary or unicode
    """
    return a.dtype.kind in ("S", "U")


def join(sep, seq, axis=None):
    """
    Note: zero-length elements will be skipped.
    """
    assert is_array(seq)

    if axis is None:
        # char Element-wise join
        return np.char.join(sep, seq)

    if isinstance(seq, np.ma.MaskedArray):
        divided = np.rollaxis(seq.filled(""), axis=axis)
    else:
        divided = np.rollaxis(seq, axis=axis)

    return functools.reduce(
        lambda a, b: np.char.rstrip(np.char.add(a, b), sep),
        more_itertools.roundrobin(divided, itertools.repeat(sep, len(divided) - 1))
    )
