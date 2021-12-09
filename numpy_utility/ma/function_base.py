import inspect

import numpy as np
import functools
from .. import core as _core_module


__all__ = [
    "is_structured_array",
    "for_masked_array",
    "array_from_nonmasked_values"
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
            return _core_module.from_dict({col: x1[col] / x2[col] for col in x1.dtype.names})
        else:
            raise NotImplementedError
    else:
        return np.divide(x1, x2)


def for_masked_array(func):
    parameter_names = list(inspect.signature(func).parameters.keys())
    first_param_name = parameter_names[0]

    @functools.wraps(func)
    def _inner(*args, **kwargs):
        if len(args) > 0:
            first_arg = args[0]
            args = args[1:]
        elif first_param_name in kwargs:
            first_arg = kwargs.pop(first_param_name)
        else:
            raise RuntimeWarning("invalid structure of func")

        if np.ma.isMaskedArray(first_arg):
            ret = func(first_arg.compressed(), *args, **kwargs)
            a = np.ma.empty(len(first_arg), dtype=np.asarray(ret).dtype)
            a.mask = True
            a[~first_arg.mask] = ret
        else:
            a = func(first_arg, *args, **kwargs)
        return a
    return _inner


def array_from_nonmasked_values(a_nonmasked, mask, dtype=None):
    mask = np.asarray(mask)
    if dtype is None and isinstance(a_nonmasked, np.ndarray):
        dtype = a_nonmasked.dtype
    a = np.ma.empty(mask.shape, dtype=dtype)
    a[~mask] = a_nonmasked
    a.mask = mask
    return a
