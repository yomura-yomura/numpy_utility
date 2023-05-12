import numpy as np
import warnings
from ..core.numerictypes import is_integer, is_floating, is_numeric
from ..core.fromnumeric import is_array

__all__ = ["histogram", "histogram_bin_edges", "histogram_bin_centers", "histogram_bin_widths"]
n_bins_limit = 10000


def histogram_bin_centers(bins):
    bins = np.asarray(bins)
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
    bins = np.asarray(bins)
    if np.issubdtype(bins.dtype, np.datetime64):
        bins = bins.astype(int)
    elif np.issubdtype(bins.dtype, np.bool_) or np.issubdtype(bins.dtype, np.str_):
        return np.array([1] * len(bins))
    elif is_numeric(bins):
        pass
    else:
        raise TypeError(f"Unexpected type: {bins.dtype}")
    return bins[1:] - bins[:-1]


def histogram_bin_edges(a, bins=10, range=None, weights=None, log=False):
    if is_array(bins):
        return bins

    a = a if isinstance(a, np.ndarray) else np.asarray(a)
    if np.issubdtype(a.dtype, np.datetime64):
        bins = histogram_bin_edges((a - np.min(a)).astype(float), bins, range, weights)

        time_unit = np.datetime_data(a.dtype)[0]
        bins = np.min(a) + bins.astype(f"timedelta64[{time_unit}]")
        return bins
    elif is_integer(a):
        pass
    #     bins = np.unique(a)
    elif is_floating(a):
        pass
    elif np.issubdtype(a.dtype, np.bool_) or np.issubdtype(a.dtype, np.str_):
        # Not implemented yet that bins affect nothing
        bins = np.unique(a)
        return bins
    else:
        raise NotImplementedError(f"Unexpected type: {a.dtype}")

    try:
        if log:
            log_a = np.log10(a)
            log_a = log_a[~np.isnan(log_a)]
            bins = 10 ** histogram_bin_edges(log_a, bins, range, weights, log=False)
        else:
            if range is None:
                range = (np.min(a), np.max(a))

            if (
                    (range[1] - range[0] < 1e7) or
                    (bins is not None)
            ):
                # print(bins)
                bins = np.histogram_bin_edges(a, bins=bins, range=range, weights=weights)
            else:
                warnings.warn(
                    "It may cause memory leak and hang"
                    f" due to significantly different between range first and second: {range}\n"
                    f"Use bin size {n_bins_limit}"
                )
                bins = np.linspace(*range, n_bins_limit)

        if bins.size > n_bins_limit:
            warnings.warn(f"Huge bin size {bins.size} -> {n_bins_limit}")
            bins = np.linspace(np.min(bins), np.max(bins), n_bins_limit)
    except MemoryError as e:
        warnings.warn(f"Encountered MemoryError: {e}")
        bins = np.linspace(np.min(a), np.max(a), n_bins_limit)

    return bins


def histogram(a, bins=10, range=None, weights=None, density=None):
    assert a.ndim == 1

    bins = histogram_bin_edges(a, bins, range, weights)

    if is_numeric(a):
        counts, bins = np.histogram(a, bins=bins, range=range, weights=weights, density=density)
    elif np.issubdtype(a.dtype, np.datetime64):
        subtracted_bins = (bins - np.min(a)).astype(int)
        subtracted_a = (a - np.min(a)).astype(int)
        counts, _ = np.histogram(a, bins=bins, range=range, weights=weights, density=density)
        time_unit = np.datetime_data(a.dtype)[0]
        bins = np.min(a) + subtracted_bins.astype(f"timedelta64[{time_unit}]")
    elif np.issubdtype(a.dtype, np.bool_) or np.issubdtype(a.dtype, np.str_):
        x, y = np.unique(a, return_counts=True)
        counts = np.zeros(len(bins), dtype="i8")
        # print(bins, x.shape, bins.shape)
        counts[np.isin(bins, x)] = y

        if density is True:
            counts = counts / np.sum(counts)
    else:
        raise NotImplementedError(f"Unexpected type: {a.dtype}")

    return counts, bins
