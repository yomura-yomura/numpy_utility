"""
functions for jagged array
"""
from ..core import is_array
import numpy as np


__all__ = ["ndim", "flatten", "apply", "from_jagged_array", "reshape"]


def _estimate_type(a):
    dtype = np.array(a).dtype
    # if np.issubdtype(dtype, np.object_):
    #     raise ValueError(f"Couldn't estimate type of {a}")
    return dtype


def ndim(a):
    a = np.array(a, dtype=object)
    if np.ndim(a) == 0:
        return 0
    elif np.ndim(a) == 1 and (len(a) == 0 or any(np.ndim(ia) == 0 for ia in a)):
        return 1
    else:
        return max(ndim(ia) for ia in a) + 1


def flatten(a, max_depth=-1):
    """
    extended for jagged array
    """

    ndim_a = ndim(a)
    if max_depth < 0:
        max_depth = ndim_a + max_depth + 1
        assert 0 <= max_depth
    elif max_depth > ndim_a:
        max_depth = ndim_a
        # raise ValueError(f"max_depth={max_depth} > ndim={ndim(a)}")

    if max_depth == 0:
        return [a]

    # if np.ndim(a) == 0:
    if ndim_a == 0:
        return [a]
    else:
        return [iia for ia in a for iia in flatten(ia, max_depth - 1)]


def apply(func, a, depth=-1, keepdims=True):
    """

    :param func:
    :param a:
    :param depth:
    :param keepdims: keeps dimensions if possible
    :return:
    """

    if depth < 0:
        depth = ndim(a) + depth + 1

    applied_flatten_a = np.array([func(ia) for ia in flatten(a, max_depth=depth)], dtype=object)

    if keepdims:
        new_shape = (*np.shape(np.asarray(a, dtype=object))[:depth], *applied_flatten_a.shape[1:])
        if np.prod(new_shape) == applied_flatten_a.size:
            return applied_flatten_a.reshape(new_shape)
    return applied_flatten_a


def from_jagged_array(pylist, horizontal_size=-1, dtype=None, axis=-1):
    assert is_array(pylist)
    assert axis == -1

    lens = apply(np.size, pylist, depth=-2)
    # lens = apply(np.size, pylist, depth=-1)

    lens_max = np.max(lens)

    mask = lens[..., np.newaxis] <= np.expand_dims(np.arange(max(lens_max, horizontal_size)), tuple(range(lens.ndim)))

    if isinstance(pylist, np.ma.MaskedArray):
        assert all(len1 == len2 for len1, len2 in zip(mask.shape, pylist.mask.shape))
        if mask.ndim > pylist.mask.ndim:
            pylist_mask = np.expand_dims(pylist.mask, axis=(-1 - np.arange(mask.ndim - pylist.mask.ndim)).tolist())
        elif mask.ndim == pylist.mask.ndim:
            pylist_mask = pylist.mask
        else:
            raise NotImplementedError

        mask |= pylist_mask
        flatten_pylist = flatten(pylist.data[~pylist.mask])
    else:
        flatten_pylist = flatten(pylist)

    if dtype is None:
        dtype = _estimate_type(flatten_pylist)

    a = np.ma.empty(mask.shape, dtype)
    a[~mask] = flatten_pylist
    a.mask = mask

    if 0 < horizontal_size < lens_max:
        return a[..., :horizontal_size]

    return a


def reshape(a, newshape):
    if np.ma.isMaskedArray(a):
        new_a = reshape(a.data, newshape)
        new_a.mask |= reshape(a.mask, newshape)
    else:
        a = np.asarray(a)
        new_a = np.ma.empty(newshape, dtype=a.dtype)
        mask = np.arange(np.prod(newshape)).reshape(newshape) >= len(a)
        new_a[~mask] = a
        new_a.mask = mask
    return new_a
