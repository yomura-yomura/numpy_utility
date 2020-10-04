import numpy as np
import inspect
import functools
import tqdm

__all__ = [
    "vectorize",
    "vectorize_wrapper"
]


@np.core.overrides.set_module("numpy_utility")
def vectorize(func, progress_kwargs={}):
    def _len(a):
        try:
            return len(a)
        except TypeError:
            return 0

    @functools.wraps(func)
    def _inner(*args, **kwargs):
        # Copy all args values into kwargs
        kwargs.update(dict(zip(inspect.signature(func).parameters.keys(), args)))

        # Expand 0-dim to corresponding n-dim args
        common_ndim = max(np.ndim(v) for v in kwargs.values())
        kwargs.update(dict([
            (k, [v] * common_ndim) if np.ndim(v) == 0 else (k, v)
            for k, v in kwargs.items()
        ]))

        kwargs_rows = [
            {k: v for k, v in zip(kwargs.keys(), row_values)}
            for row_values in zip(*np.broadcast_arrays(*kwargs.values()))
        ]

        # print(kwargs_rows)
        if progress_kwargs:
            kwargs_rows = tqdm.tqdm(kwargs_rows, **progress_kwargs)
        returned_rows = [func(**kwargs_row) for kwargs_row in kwargs_rows]

        common_returned_length = _len(returned_rows[0])
        print(common_returned_length)
        assert all(common_returned_length == _len(returned_row) for returned_row in returned_rows)

        if common_returned_length == 0:
            return returned_rows
        else:
            return (
                np.ma.concatenate(returned_col)
                for returned_col in zip(*returned_rows)
            )

    return _inner


@np.core.overrides.set_module("numpy_utility")
def vectorize_wrapper(progress_kwargs={}):
    def _inner(func):
        return vectorize(func, progress_kwargs)
    return _inner
