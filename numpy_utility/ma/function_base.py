import numpy as np
import functools
import itertools
from ..core import is_array


__all__ = [
    "from_jagged_array"
    # "vectorize"
]


def _estimate_type(a):
    dtype = np.array(a).dtype
    if np.issubdtype(dtype, np.object_):
        raise ValueError(f"Couldn't estimate type of {a}")
    return dtype


def flatten(a, max_depth=-1):
    """
    extended for jagged array
    """
    if max_depth == 0:
        return [a]

    if np.ndim(a) == 0:
        return [a]
    else:
        return [iia for ia in a for iia in flatten(ia, max_depth - 1)]


def apply(func, a, axis=None):
    assert axis is None
    applied_flatten_a = [func(ia) for ia in flatten(a, max_depth=np.ndim(a))]
    return np.reshape(applied_flatten_a, (*np.shape(a), *np.shape(applied_flatten_a)[1:]))


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


def from_jagged_array(pylist, horizontal_size=-1, dtype=None):
    # assert np.ndim(pylist) < 2
    assert is_array(pylist)

    # lens = np.array([len(pl) for pl in pylist])
    lens = apply(len, pylist)
    lens_max = lens.max()
    # flatten_pylist = [e for pl in pylist for e in pl]
    flatten_pylist = flatten(pylist)

    if dtype is None:
        dtype = _estimate_type(flatten_pylist)

    # mask = lens[:, np.newaxis] <= np.arange(max(lens_max, horizontal_size))
    mask = lens[..., np.newaxis] <= np.expand_dims(np.arange(max(lens_max, horizontal_size)),
                                                   np.arange(lens.ndim).tolist())
    mask = lens <= np.expand_dims(np.arange(max(lens_max, horizontal_size)), np.arange(lens.ndim - 1).tolist())
    a = np.ma.empty(mask.shape, dtype)
    a[~mask] = flatten_pylist
    a.mask = mask

    if 0 < horizontal_size < lens_max:
        return a[..., :horizontal_size]

    return a


# 車輪の再発明: np.vectorizeで実現可能
# def vectorize(pyfunc, otypes=None, doc=None, excluded=None,
#               cache=False, signature=None):
#     vectorized_func = np.vectorize(pyfunc, otypes, doc, excluded, cache, signature)
#
#     if otypes is not None and 1 < len(otypes):
#         multiple_ret = True
#     else:
#         multiple_ret = False
#
#     @functools.wraps(pyfunc)
#     def _inner(*args, **kwargs):
#         masks = [
#             arg.mask
#             for arg in itertools.chain(args, kwargs.values())
#             if isinstance(arg, np.ma.MaskedArray)
#         ]
#
#         # Pass non-masked-arguments to normal np.vectorize
#         if len(masks) == 0:
#             return vectorized_func(*args, **kwargs)
#
#         mask = np.sum(masks, axis=0) != 0
#
#         args = (arg[mask] for arg in args)
#         kwargs = {k: v[mask] for k, v in kwargs.items()}
#         partial_ret = vectorized_func(*args, **kwargs)
#
#         def f(partial_returned):
#             ret = np.ma.empty(mask.shape, dtype=partial_returned.dtype)
#             ret.mask = True
#             ret[~mask] = partial_returned
#             return ret
#
#         if multiple_ret:
#             return tuple(
#                 f(pr)
#                 for pr in partial_ret
#             )
#         else:
#             return f(partial_ret)
#
#     return _inner
