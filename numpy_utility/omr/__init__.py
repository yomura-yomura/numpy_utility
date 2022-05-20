import numpy as np

# 分類難しいの。とりあえず。


__all__ = ["binning", "get_indices_groups_of_continuous_duplicate_numbers"]


def binning(x, y, min_bin_x=None, max_bin_x=None, step_bin_x=1, sqrt_n_division=True, allowed_minimum_size=None,
            return_sample_size=False):
    """
    x, y : array-like object, MaskedArray
    """
    if isinstance(x, np.ma.MaskedArray):
        x = x.compressed()
    if isinstance(y, np.ma.MaskedArray):
        y = y.compressed()

    sorted_x = np.sort(x)
    sorted_y = y[np.argsort(x)]

    _binned_x = np.arange(
        np.min(x) if min_bin_x is None else min_bin_x,
        np.max(x) if max_bin_x is None else max_bin_x,
        step_bin_x
    )
    _binned_y = [
        sorted_y[(lower_x < sorted_x) & (sorted_x < upper_x)]
        for lower_x, upper_x in zip(_binned_x[:-1], _binned_x[1:])
    ]

    binned_x = _binned_x[:-1] + step_bin_x / 2

    if allowed_minimum_size is not None:
        binned_x = binned_x[[True if allowed_minimum_size <= y.size else False for y in _binned_y]]
        _binned_y = [y for y in _binned_y if allowed_minimum_size <= y.size]

    n_y = np.array([y.size if y.size != 0 else 0 for y in _binned_y])
    mean_y = np.array([y.mean() if y.size != 0 else 0 for y in _binned_y])
    err_y = np.array([(y - my).std() if y.size != 0 else 0 for my, y in zip(mean_y, _binned_y)])

    if sqrt_n_division:
        err_y = err_y / np.sqrt(n_y)

    ret = (binned_x, mean_y, err_y)

    if return_sample_size:
        ret += (n_y,)

    return ret


def get_indices_groups_of_continuous_duplicate_numbers(x):
    from itertools import groupby
    from operator import itemgetter

    for k, g in groupby(enumerate(x), lambda ix: ix[0] - ix[1]):
        yield list(map(itemgetter(1), g))
