import numpy as np
import functools
import itertools
from ..core import from_dict


__all__ = [
    "is_structured_array",
    "merge_arrays"
]


def is_structured_array(a: np.ndarray):
    return a.dtype.names is not None


def compressed(x: np.ma.MaskedArray):
    if is_structured_array(x):
        raise NotImplementedError

    compressed_x = x.compressed()

    return compressed_x


def divide(x1, x2):
    if x1.dtype.names is not None or x2.dtype.names is not None:
        if x1.dtype == x2.dtype:
            from ..core import from_dict
            return from_dict({col: x1[col] / x2[col] for col in x1.dtype.names})
        else:
            raise NotImplementedError
    else:
        return np.divide(x1, x2)


def merge_arrays(arrays):
    raw_dict = [(name, a[name]) for a in arrays for name in a.dtype.names]
    if any(np.unique([k for k, _ in raw_dict], return_counts=True)[1] > 1):
        n, c = np.unique([k for k, _ in raw_dict], return_counts=True)
        raise ValueError(f"field '{n[np.argmax(c)]}' occurs more than once")

    return from_dict(dict(raw_dict))


