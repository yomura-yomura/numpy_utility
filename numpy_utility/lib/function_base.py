import numpy as np
import inspect
import functools
import tqdm
import sys


__all__ = [
    "vectorize",
    "vectorize_wrapper"
]


def vectorize(func, progress_kwargs={}, multi_output=False, errors="raise", return_as_numpy_array=True, ignore_kwargs=()):
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
        # Add all args values to kwargs
        kwargs.update(dict(zip(inspect.signature(func).parameters.keys(), args)))

        # Expand 0-dim to corresponding n-dim args
        _common_ndim = max(np.ndim(v) for k, v in kwargs.items() if k not in ignore_kwargs)
        # 0-dim to 1-dim args
        common_ndim = max(_common_ndim, 1)
        assert common_ndim == 1

        kwargs.update(dict([
            (k, [v] * common_ndim) if (k in ignore_kwargs) or np.ndim(v) == 0 else (k, v)
            for k, v in kwargs.items()
        ]))

        from .. import ja
        common_len = ja.apply(len, list(kwargs.values()), 1).max()

        kwargs_rows = [
            {k: v for k, v in zip(kwargs.keys(), row_values)}
            for row_values in zip(*(v if len(v) == common_len else v * common_len for v in kwargs.values()))
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

        if return_as_numpy_array:
            if np.any([isinstance(r, np.ma.MaskedArray) for row in returned_rows for r in row]):
                def array(a):
                    return np.ma.MaskedArray([e.data for e in a], [e.mask for e in a])
            else:
                array = np.array
            ret = [array(list(returned_col)) for returned_col in zip(*returned_rows)]
        else:
            ret = [list(returned_col) for returned_col in zip(*returned_rows)]

        if mask.any():
            assert return_as_numpy_array == np.True_
            for i in range(len(ret)):
                new = np.ma.empty((mask.shape[0], *ret[i].shape[1:]), dtype=ret[i].dtype)
                new.mask = mask
                new[~mask] = ret[i]
                ret[i] = new

        if _common_ndim == 0:
            assert all(len(col) == 1 for col in ret)
            ret = [col[0] for col in ret]

        if multi_output:
            return ret
        else:
            assert len(ret) == 1
            return ret[0]

    return _inner


def vectorize_wrapper(progress_kwargs={}, multi_output=False, errors="raise", return_as_numpy_array=True, ignore_kwargs=()):
    def _inner(func):
        return vectorize(func, progress_kwargs, multi_output, errors, return_as_numpy_array, ignore_kwargs)
    return _inner
