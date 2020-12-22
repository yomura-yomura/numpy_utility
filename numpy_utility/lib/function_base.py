import numpy as np
import inspect
import functools
import tqdm
import sys


__all__ = [
    "vectorize",
    "vectorize_wrapper"
]


# @np.core.overrides.set_module("numpy_utility")
def vectorize(func, progress_kwargs={}, multi_output=False, concat=False, errors="raise"):
    def _len(a):
        try:
            return len(a)
        except TypeError:
            return 0

    if errors == "raise":
        func_ = func
    elif errors == "coerce":
        def func_(**kwargs):
            try:
                return func(**kwargs)
            except BaseException as e:
                print(repr(e), file=sys.stderr)
                return None
    else:
        raise ValueError(errors)

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

        returned_rows = [func_(**kwargs_row) for kwargs_row in kwargs_rows]
        # Remove invalid rows
        mask = np.array([r is None for r in returned_rows])
        returned_rows = [r for r in returned_rows if r is not None]

        if not multi_output:
            returned_rows = [[r] for r in returned_rows]

        if len(returned_rows) == 0:
            return None
            # raise ValueError("No returned values are not allowed.")

        common_returned_length = _len(returned_rows[0])
        assert all(common_returned_length == _len(returned_row) for returned_row in returned_rows)

        if concat:
            if np.any([isinstance(r, np.ma.MaskedArray) for row in returned_rows for r in row]):
                concatenate = np.ma.concatenate
            else:
                concatenate = np.concatenate
            ret = [concatenate(returned_col) for returned_col in zip(*returned_rows)]
        else:
            if np.any([isinstance(r, np.ma.MaskedArray) for row in returned_rows for r in row]):
                def array(a):
                    return np.ma.MaskedArray([e.data for e in a], [e.mask for e in a])
            else:
                array = np.array
            ret = [array(list(returned_col)) for returned_col in zip(*returned_rows)]

        if mask.any():
            for i in range(len(ret)):
                new = np.ma.empty((mask.shape[0], *ret[i].shape[1:]), dtype=ret[i].dtype)
                new.mask = mask
                new[~mask] = ret[i]
                ret[i] = new

        if multi_output:
            return ret
        else:
            assert len(ret) == 1
            return ret[0]

    return _inner


@np.core.overrides.set_module("numpy_utility")
def vectorize_wrapper(progress_kwargs={}, multi_output=False, concat=False, errors="raise"):
    def _inner(func):
        return vectorize(func, progress_kwargs, multi_output, concat, errors)
    return _inner
