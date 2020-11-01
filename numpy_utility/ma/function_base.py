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


def from_jagged_array(pylist, horizontal_size=-1):
    assert is_array(pylist)

    lens = np.array([len(pl) for pl in pylist])
    lens_max = lens.max()
    flatten_pylist = [e for pl in pylist for e in pl]

    mask = lens[:, np.newaxis] <= np.arange(max(lens_max, horizontal_size))
    a = np.ma.empty(mask.shape, _estimate_type(flatten_pylist))
    a[~mask] = flatten_pylist
    a.mask = mask

    if 0 < horizontal_size < lens_max:
        return a[:, :horizontal_size]

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
