import numpy as np
import functools
import more_itertools
import itertools
from ..core import is_array

__all__ = ["join", "read_null_terminated_string"]


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

    if isinstance(seq, np.ma.MaskedArray):
        # print(seq)
        # raise NotImplementedError
        pass
    else:
        seq = np.asarray(seq)

    if axis is None:
        # char Element-wise join
        return np.char.join(sep, seq)

    if isinstance(seq, np.ma.MaskedArray):
        divided = np.rollaxis(seq.filled(""), axis=axis)
    else:
        divided = np.rollaxis(seq, axis=axis)

    return functools.reduce(
        lambda a, b: np.char.add(a, b),
        more_itertools.roundrobin(divided, itertools.repeat(sep, len(divided) - 1))
    )


def read_null_terminated_string(buffer):
    s_itemsize = buffer.dtype.itemsize
    null_terminated_string = np.frombuffer(buffer, dtype="S1").reshape(-1, s_itemsize)
    assert (null_terminated_string == b"").any(axis=1).all()
    position_null_terminated = np.argmax(null_terminated_string == b"", axis=1)
    invalid_mask = position_null_terminated[:, np.newaxis] < np.arange(16)
    null_terminated_string[invalid_mask] = b""
    return np.frombuffer(null_terminated_string, dtype=f"S{s_itemsize}")
