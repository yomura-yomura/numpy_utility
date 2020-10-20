import numpy as np
import inspect
import functools
import tqdm

__all__ = [
    "vectorize",
    "vectorize_wrapper"
]


@np.core.overrides.set_module("numpy_utility")
def vectorize(func, progress_kwargs={}, multi_output=False):
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
        # 0-dim to 1-dim args
        common_ndim = max(common_ndim, 1)

        kwargs.update(dict([
            (k, [v] * common_ndim) if np.ndim(v) == 0 else (k, v)
            for k, v in kwargs.items()
        ]))

        kwargs_rows = [
            {k: v for k, v in zip(kwargs.keys(), row_values)}
            for row_values in zip(*np.broadcast_arrays(*kwargs.values()))
        ]

        if progress_kwargs:
            kwargs_rows = tqdm.tqdm(kwargs_rows, **progress_kwargs)

        if multi_output:
            returned_rows = [func(**kwargs_row) for kwargs_row in kwargs_rows]
        else:
            returned_rows = [[func(**kwargs_row)] for kwargs_row in kwargs_rows]

        common_returned_length = _len(returned_rows[0])
        assert all(common_returned_length == _len(returned_row) for returned_row in returned_rows)

        # if common_returned_length == 0:
        #     return returned_rows
        # else:
        if np.any([isinstance(row, np.ma.MaskedArray) for row in returned_rows]):
            concatenate = np.ma.concatenate
        else:
            concatenate = np.concatenate

        ret = [concatenate(returned_col) for returned_col in zip(*returned_rows)]
        if multi_output:
            return ret
        else:
            assert len(ret) == 1
            return ret[0]

    return _inner


@np.core.overrides.set_module("numpy_utility")
def vectorize_wrapper(progress_kwargs={}, multi_output=False):
    def _inner(func):
        return vectorize(func, progress_kwargs, multi_output)
    return _inner
