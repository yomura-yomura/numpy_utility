import numpy as np
import functools
import itertools


__all__ = [
    # "vectorize"
]


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
