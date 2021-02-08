"""
functions for jagged array
"""
from ..core import is_array
import numpy as np


__all__ = ["ndim", "flatten", "apply", "from_jagged_array"]


def _estimate_type(a):
    dtype = np.array(a).dtype
    if np.issubdtype(dtype, np.object_):
        raise ValueError(f"Couldn't estimate type of {a}")
    return dtype


def ndim(a):
    if np.ndim(a) == 0 or len(a) == 0:
        return 0
    elif np.ndim(a) == 1 and any(np.ndim(ia) == 0 for ia in a):
        return 1
    else:
        return max(ndim(ia) for ia in a) + 1


def flatten(a, max_depth=-1):
    """
    extended for jagged array
    """
    if max_depth < 0:
        max_depth = ndim(a) + max_depth + 1
        assert 0 <= max_depth
    elif max_depth > ndim(a):
        raise ValueError(max_depth)

    if max_depth == 0:
        return [a]

    if np.ndim(a) == 0:
        return [a]
    else:
        return [iia for ia in a for iia in flatten(ia, max_depth - 1)]


def apply(func, a, depth=-1):
    if depth < 0:
        depth = ndim(a) + depth

    applied_flatten_a = [func(ia) for ia in flatten(a, max_depth=depth)]
    return np.reshape(applied_flatten_a, (*np.shape(a)[:depth], *np.shape(applied_flatten_a)[1:]))


def from_jagged_array(pylist, horizontal_size=-1, dtype=None, axis=-1):
    assert is_array(pylist)
    assert axis == -1

    lens = apply(len, pylist, depth=-1)

    lens_max = np.max(lens)
    flatten_pylist = flatten(pylist)

    if dtype is None:
        dtype = _estimate_type(flatten_pylist)

    mask = lens[..., np.newaxis] <= np.expand_dims(np.arange(max(lens_max, horizontal_size)), tuple(range(lens.ndim)))
    a = np.ma.empty(mask.shape, dtype)
    a[~mask] = flatten_pylist
    a.mask = mask

    if 0 < horizontal_size < lens_max:
        return a[..., :horizontal_size]

    return a