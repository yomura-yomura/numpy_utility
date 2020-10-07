import numpy as np
import warnings
from ..core.numerictypes import is_integer, is_floating, is_numeric
from ..core.fromnumeric import is_array


__all__ = ["histogram", "histogram_bin_edges", "histogram_bin_centers", "histogram_bin_widths"]


def histogram_bin_centers(bins):
    if np.issubdtype(bins.dtype, np.datetime64):
        time_unit = np.datetime_data(bins.dtype)[0]
        return histogram_bin_centers(bins.astype(int)).astype(f"datetime64[{time_unit}]")
    elif np.issubdtype(bins.dtype, np.bool_) or np.issubdtype(bins.dtype, np.str_):
        return bins
    elif is_numeric(bins):
        return (bins[1:] + bins[:-1]) * 0.5
    else:
        raise NotImplementedError


def histogram_bin_widths(bins):
    if np.issubdtype(bins.dtype, np.datetime64):
        bins = bins.astype(int)
    elif np.issubdtype(bins.dtype, np.bool_) or np.issubdtype(bins.dtype, np.str_):
        return np.array([1] * len(bins))
    elif is_numeric(bins):
        pass
    else:
        raise NotImplementedError
    return bins[1:] - bins[:-1]


def histogram_bin_edges(a, bins=10, range=None, weights=None):
    n_bins_limit = 10000

    if is_array(bins):
        return bins

    if np.issubdtype(a.dtype, np.datetime64):
        if np.issubdtype(np.array(bins).dtype, np.datetime64):
            bins = (bins - np.min(bins)).astype(int)
        if is_array(bins):
            bins[:-1] = np.floor(bins[:-1])
            bins[-1] = np.ceil(bins[-1])
        bins = histogram_bin_edges((a - np.min(a)).astype(int), bins, range, weights)
        time_unit = np.datetime_data(a.dtype)[0]
        bins = np.min(a) + bins.astype(f"timedelta64[{time_unit}]")
        return bins
    elif is_integer(a):
        bins = np.unique(a)
    elif is_floating(a):
        pass
    elif np.issubdtype(a.dtype, np.bool_) or np.issubdtype(a.dtype, np.str_):
        bins = np.unique(a)
    else:
        raise NotImplementedError(f"Unexpected type: {a.dtype}")

    try:
        bins = np.histogram_bin_edges(a, bins, range, weights)
        if bins.size > n_bins_limit:
            warnings.warn(f"Huge bin size {bins.size} -> {n_bins_limit}")
            bins = np.linspace(np.min(bins), np.max(bins), n_bins_limit)
    except MemoryError as e:
        warnings.warn(f"Encountered MemoryError: {e}")
        bins = np.linspace(np.min(bins), np.max(bins), n_bins_limit)

    return bins


def histogram(a, bins=10, range=None, weights=None, density=None):
    normed = None

    a = np.array(a)
    assert a.ndim == 1

    bins = histogram_bin_edges(a, bins, range, weights)

    if is_numeric(a):
        counts, bins = np.histogram(a, bins, range, normed, weights, density)
    elif np.issubdtype(a.dtype, np.datetime64):
        subtracted_bins = (bins - np.min(a)).astype(int)
        subtracted_a = (a - np.min(a)).astype(int)
        counts, _ = np.histogram(subtracted_a, subtracted_bins, range, normed, weights, density)
        time_unit = np.datetime_data(a.dtype)[0]
        bins = np.min(a) + bins.astype(f"timedelta64[{time_unit}]")
    elif np.issubdtype(a.dtype, np.bool_) or np.issubdtype(a.dtype, np.str_):
        bins, counts = np.unique(a, return_counts=True)
        if density is True:
            counts = counts / np.sum(counts)
    else:
        raise NotImplementedError(f"Unexpected type: {a.dtype}")

    return counts, bins

