import numpy as np
import numpy.lib.recfunctions
from ..core import is_integer

__all__ = [
    "get_array_matched_with_boolean_array",
    "is_array",
    "combine_structured_arrays",
    "add_new_field_to",
    "remove_field_from",
    "change_field_format_to",
    "get_new_array_with_field_names",
    "search_matched",
    "from_dict",
    "reshape",
    "any_along_column",
    "all_along_column",
    "flatten_structured_array",
    "groupby"
]


def get_array_matched_with_boolean_array(a, boolean_array, remove_all_masked_rows=False):
    assert(1 <= boolean_array.ndim <= 2)
    assert(a.size == boolean_array.shape[-1])

    new_array = np.ma.empty(boolean_array.shape, dtype=a.dtype)
    new_array[boolean_array] = a[np.where(boolean_array)[-1]]
    new_array.mask = ~boolean_array

    if remove_all_masked_rows:
        assert(new_array.ndim >= 2)
        new_array = new_array[~np.all(new_array.mask.view(bool), axis=-1)]

    return new_array


def is_array(obj):
    return (
        isinstance(obj, (list, tuple, set)) or
        (isinstance(obj, np.ndarray) and np.ndim(obj) != 0)
    )


def combine_structured_arrays(a1, a2):
    assert a1.shape == a2.shape
    assert np.isin(a1.dtype.names, a2.dtype.names).any() == False
    new_fields = [(name, *sub) for name, *sub in a2.dtype.descr if name != ""]
    return add_new_field_to(a1, new_fields, a2)


def add_new_field_to(a, new_field, filled=None):
    if is_array(new_field):
        if isinstance(new_field, tuple):
            new_field = [new_field]
        new_a = np.zeros(a.shape, a.dtype.descr + new_field)
        np.lib.recfunctions.recursive_fill_fields(a, new_a)
        if filled is not None:
            for name, *_ in new_field:
                new_a[name] = filled
        return new_a
    elif isinstance(new_field, str):
        if filled is None:
            raise ValueError
        filled_a = np.array(filled)
        return add_new_field_to(a, [(new_field, filled_a.dtype.descr)], filled_a)
    else:
        raise ValueError(new_field)


def remove_field_from(a, field):
    if is_array(field):
        return get_new_array_with_field_names(
            a, [field_name for field_name, *_ in a.dtype.descr if field_name not in field]
        )
    else:
        return get_new_array_with_field_names(
            a, [field_name for field_name, *_ in a.dtype.descr if field_name != field]
        )


def change_field_format_to(a, new_field_format):
    """
    new_field_format: dict type: {[field name]: [format]}
    """
    new_type = [(k, *sub) if k not in new_field_format.keys() else (k, new_field_format[k], *sub[1:])
                for k, *sub in a.dtype.descr]
    new_a = np.empty_like(a, new_type)
    np.lib.recfunctions.recursive_fill_fields(a, new_a)
    return new_a


def get_new_array_with_field_names(a, field_names):
    assert is_array(field_names)
    new_a = a[field_names]
    new_a = new_a.astype([d for d in new_a.dtype.descr if d[0] != ""])
    return new_a


def search_matched(a, v):
    sorted_indices = np.argsort(a)
    indices = np.searchsorted(a, v, side="left", sorter=sorted_indices)
    indices_right = np.searchsorted(a, v, side="right", sorter=sorted_indices)

    spans = indices_right - indices
    if np.any(spans > 1):
        assert np.any(np.unique(v, return_counts=True)[1] > 1)
        raise ValueError("duplicate values found in v")

    # if np.any(spans == 0):
    #     return np.ma.MaskedArray(data=indices, mask=spans == 0)
    # else:
    return indices[spans == 1]


# Maybe should be in a file _multiarray_umath.py
def from_dict(data, strict=True, use_common_dims=True):
    if isinstance(data, dict):
        new_array = [from_dict(v, strict, use_common_dims) for v in data.values()]
        masked_found = any(isinstance(na, np.ma.MaskedArray) for na in new_array)
    elif isinstance(data, np.ndarray):
        return data
    else:
        return np.array(data)

    if use_common_dims:
        max_ndim = min(v.ndim for v in new_array)
        ndim_start = np.count_nonzero([
            len(np.unique([v.shape[:i] for v in new_array], axis=0)) == 1 for i in range(1, max_ndim + 1)
        ])
        if ndim_start > 0:
            common_shape = np.unique([v.shape[:ndim_start] for v in new_array], axis=0)[0]
        else:
            common_shape = ()

        if len(common_shape) > 1:
            flatten_ret = from_dict({
                k: v.flatten() if v.ndim <= ndim_start else v.reshape((-1, *v.shape[ndim_start:]))
                for k, v in data.items()
            }, strict, use_common_dims)
            return flatten_ret.reshape(common_shape)
    else:
        ndim_start = 0

    def get_dtype(a):
        pre_descr = a.dtype.descr[0][1:] if a.dtype.names is None else [a.dtype.descr]
        if a.ndim <= ndim_start:
            return pre_descr
        else:
            return (*pre_descr, a.shape[ndim_start:])

    new_dtype = [(k, *get_dtype(na)) for k, na in zip(data.keys(), new_array)]

    if strict is True:
        if not all(len(new_array[0]) == len(na) for na in new_array[1:]):
            raise ValueError(f"Mismatch length between {data.keys()}")

    if masked_found:
        return np.ma.mrecords.fromarrays(
            new_array,
            dtype=new_dtype
        ).view(np.ma.MaskedArray)
    else:
        return np.array(list(zip(*new_array)), new_dtype)


def reshape(a, newshape, drop=True):
    # if drop is False:
    #     return np.reshape(a, newshape)

    if is_array(newshape):
        pass
    elif is_integer(newshape):
        newshape = (newshape,)
    else:
        raise TypeError(f"'{type(newshape)}' object cannot be interpreted as an integer")

    a = np.array(a)
    newshape = np.array(newshape)

    i_unknown_dimensions = np.where(newshape < 0)[0]
    if i_unknown_dimensions.size == 0:
        pass
    elif i_unknown_dimensions.size == 1:
        newshape[i_unknown_dimensions[0]] = np.floor(a.size / np.prod(newshape[newshape >= 0]))
    else:
        raise ValueError("can only specify one unknown dimension")

    new_size = a.size - a.size % np.prod(newshape)
    if a.size < new_size:
        raise ValueError(f"cannot reshape array of size {a.size} into shape {tuple(newshape)}")

    if drop is True:
        return a[:new_size].reshape(newshape)
    else:
        return a.reshape(newshape)


def any_along_column(a):
    assert a.dtype.names is not None
    assert len(a.dtype.names) > 0

    ndim = a.ndim
    return np.any(
        [(a[n] if a[n].dtype.names is None else any_along_column(a[n]))
         if a[n].ndim <= ndim else a[n].any(axis=tuple(np.arange(ndim, a[n].ndim)))
         for n in a.dtype.names],
        axis=0
    )


def all_along_column(a):
    assert a.dtype.names is not None
    assert len(a.dtype.names) > 0

    ndim = a.ndim
    return np.all(
        [(a[n] if a[n].dtype.names is None else all_along_column(a[n]))
         if a[n].ndim <= ndim else a[n].all(axis=tuple(np.arange(ndim, a[n].ndim)))
         for n in a.dtype.names],
        axis=0
    )


def flatten_structured_array(a, sep="/", flatten_2d=False):
    if a.dtype.names is None:
        return a

    d = {}
    for name in a.dtype.names:
        if a[name].dtype.names is None:
            if flatten_2d is True and a[name].ndim == 2:
                for i in range(a[name].shape[1]):
                    d[f"{name}{sep}f{i}"] = a[name][:, i]
            else:
                d[name] = a[name]
        else:
            flatten = flatten_structured_array(a[name], sep, flatten_2d)
            for k in flatten.dtype.names:
                d[f"{name}{sep}{k}"] = flatten[k]
    return from_dict(d)


def groupby(a, by, *other_by, without_masked=True):
    assert is_array(a)

    def get_boolean_groupby(a, by, *other_by, without_masked=True):
        if not is_array(by):
            by = a[by]
        if len(other_by) == 0:
            return [
                (key, key == by)
                for key in np.unique(
                    by.compressed() if isinstance(by, np.ma.MaskedArray) and without_masked else by
                )
            ]
        else:
            return get_boolean_groupby(a[by], *other_by, without_masked=without_masked)

    for key, boolean_array in get_boolean_groupby(a, by, *other_by, without_masked=without_masked):
        yield (
            key,
            a[boolean_array].data if isinstance(a, np.ma.MaskedArray) and without_masked else a[boolean_array]
        )
