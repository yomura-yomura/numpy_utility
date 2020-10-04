import numpy as np
import warnings
from ..core.numerictypes import is_integer, is_floating, is_numeric


__all__ = ["histogram", "histogram_bin_edges"]


def histogram_bin_edges(a, bins=10, range=None, weights=None):
    if np.issubdtype(a.dtype, np.datetime64):
        if np.issubdtype(np.array(bins).dtype, np.datetime64):
            bins = (bins - np.min(bins)).astype(int)
        a = (a - np.min(a)).astype(int)
        bins[:-1] = np.floor(bins[:-1])
        bins[-1] = np.ceil(bins[-1])

    if is_integer(a):
        bins = np.unique(a)
    elif is_floating(a):
        pass
    elif np.issubdtype(a.dtype, np.bool_) or np.issubdtype(a.dtype, np.str_):
        bins = np.unique(a)
    else:
        raise NotImplementedError(f"Unexpected type: {a.dtype}")

    try:
        bins = np.histogram_bin_edges(a, bins, range, weights)
        if bins.size > 1000:
            warnings.warn(f"Huge bin size {bins.size} -> {1000}")
            bins = np.linspace(np.min(bins), np.max(bins), 1000)
    except MemoryError as e:
        warnings.warn(f"Encountered MemoryError: {e}")
        bins = np.linspace(np.min(bins), np.max(bins), 1000)

    return bins


def histogram(a, bins=10, range=None, weights=None, density=None):
    normed = None

    a = np.array(a)
    assert a.ndim == 1

    bins = histogram_bin_edges(a, bins, range, weights)

    if is_numeric(a):
        counts, bins = np.histogram(a, bins, range, normed, weights, density)
        width = bins[1:] - bins[:-1]
    elif np.issubdtype(a.dtype, np.datetime64):
        subtracted_a = (a - np.min(a)).astype(int)
        counts, bins = np.histogram(subtracted_a, bins, range, normed, weights, density)
        time_unit = np.datetime_data(a.dtype)[0]
        bins = np.min(a) + bins.astype(f"timedelta64[{time_unit}]")
        width = bins[1:] - bins[:-1]
    elif np.issubdtype(a.dtype, np.bool_) or np.issubdtype(a.dtype, np.str_):
        bins, counts = np.unique(a, return_counts=True)
        width = np.array([1] * len(bins))
        if density is True:
            counts = counts / np.sum(counts)
    else:
        raise NotImplementedError(f"Unexpected type: {a.dtype}")

    return counts, bins, width