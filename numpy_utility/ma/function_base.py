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
