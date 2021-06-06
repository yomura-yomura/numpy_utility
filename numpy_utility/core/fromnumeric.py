import numpy as np
import numpy.lib.recfunctions
from ..core import is_integer
import builtins
import warnings


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
    "flatten_structured_array",
    "groupby_given",
    "groupby",
    "any",
    "all",
    "sum"
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


def add_new_field_to(a, new_field, filled=None, insert_to=None):
    if is_array(new_field):
        if isinstance(new_field, tuple):
            new_field = [new_field]
        if insert_to is None:
            new_descr = a.dtype.descr + new_field
        else:
            new_descr = a.dtype.descr.copy()
            new_descr.insert(insert_to, new_field[0])

        if isinstance(a, np.ma.MaskedArray):
            new_a = np.ma.zeros(a.shape, new_descr)
            from .. import bugfix
            bugfix.np_ma_nat_fill_value.fix(new_a)
        else:
            new_a = np.zeros(a.shape, new_descr)

        np.lib.recfunctions.recursive_fill_fields(a, new_a)

        if filled is not None:
            for name, *_ in new_field:
                new_a[name] = filled
        return new_a
    elif isinstance(new_field, str):
        if filled is None:
            raise ValueError("filled is None")
        if not isinstance(filled, np.ma.MaskedArray):
            filled = np.asarray(filled)
        if filled.dtype.names is None:
            return add_new_field_to(a, (new_field, *filled.dtype.descr[0][1:]), filled, insert_to)
        else:
            return add_new_field_to(a, [(new_field, filled.dtype.descr)], filled, insert_to)
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
        raise ValueError("duplicate values found in a")

    if indices.ndim == 0:
        if spans == 1:
            return sorted_indices[indices]
        else:
            return np.array([], dtype=np.int64)
    else:
        return sorted_indices.take(indices[spans == 1])


# Maybe should be in a file _multiarray_umath.py
def from_dict(data, strict=True, use_common_dims=True):
    if isinstance(data, dict):
        new_array = [from_dict(v, strict, use_common_dims) for v in data.values()]
        masked_found = builtins.any(isinstance(na, np.ma.MaskedArray) for na in new_array)
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
        if not builtins.all(len(new_array[0]) == len(na) for na in new_array[1:]):
            raise ValueError(f"Mismatch length between {data.keys()}")

    if masked_found:
        from .. import bugfix
        ret = np.ma.mrecords.fromarrays(
            new_array,
            dtype=new_dtype
        ).view(np.ma.MaskedArray)
        bugfix.np_ma_nat_fill_value.fix(ret)
        return ret
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


class GroupBy:
        def __init__(self, a, unique_keys, matched_columns, iter_with=None):
            self.a = a
            self.unique_keys = unique_keys
            self.matched_columns = matched_columns
            self.iter_with = iter_with
            if iter_with is not None:
                assert builtins.all(len(item) == len(a) for item in iter_with)

        def __iter__(self):
            g = ((key, np.isin(self.matched_columns, key)) for key in self.unique_keys)
            if self.iter_with is None:
                return (
                    (key, self.a[boolean_array])
                    for key, boolean_array in g
                )
            else:
                return (
                    (key, self.a[boolean_array], *(item[boolean_array] for item in self.iter_with))
                    for key, boolean_array in g
                )

        def __len__(self):
            return len(self.unique_keys)


def groupby_given(a, by_given, sort=False):
    assert is_array(a)
    assert is_array(by_given)

    if sort:
        unique_keys = np.unique(by_given, axis=0)
    else:
        unique_keys, indices = np.unique(by_given, return_index=True, axis=0)
        unique_keys = unique_keys[np.argsort(indices)]

    return GroupBy(a, unique_keys, by_given)


def groupby(a, by, *nested_by, without_masked=True, iter_with=None, sort=False):
    assert is_array(a)

    def get_boolean_groupby(a, by, *nested_by, without_masked=True):
        if len(nested_by) == 0:
            if isinstance(a, np.ma.MaskedArray) and without_masked:
                if is_array(by):
                    a_by = a[by].data[~any(a[by].mask, axis="column")]
                else:
                    a_by = a[by].compressed()
            else:
                a_by = a[by]

            if sort:
                unique_keys = np.unique(a_by, axis=0)
            else:
                unique_keys, indices = np.unique(a_by, return_index=True, axis=0)
                unique_keys = unique_keys[np.argsort(indices)]

            return unique_keys, a[by]
        else:
            return get_boolean_groupby(a[by], *nested_by, without_masked=without_masked)

    return GroupBy(a, *get_boolean_groupby(a, by, *nested_by, without_masked=without_masked), iter_with)


def _along_column(func, a):
    return [
        (a[n] if a[n].dtype.names is None else func(a[n], axis="column"))
        if a[n].ndim <= a.ndim else func(a[n], axis=tuple(range(a.ndim, a[n].ndim)))
        for n in a.dtype.names
    ]


def any(a, axis=None, out=None, keepdims=np._NoValue):
    if axis == "column":
        if a.dtype.names is None:
            raise ValueError(f"'a' has no dtype.names with axis='column'")
        ret = np.any(_along_column(any, a), axis=0, out=out, keepdims=keepdims)
    else:
        ret = np.any(a, axis, out, keepdims)
    if np.issubdtype(ret.dtype, np.object_):
        raise NotImplementedError("type of returned is object")
    return ret


def all(a, axis=None, out=None, keepdims=np._NoValue):
    if axis == "column":
        if a.dtype.names is None:
            raise ValueError(f"'a' has no dtype.names with axis='column'")
        ret = np.all(_along_column(all, a), axis=0, out=out, keepdims=keepdims)
    else:
        ret = np.all(a, axis, out, keepdims)
    if np.issubdtype(ret.dtype, np.object_):
        raise NotImplementedError("type of returned is object")
    return ret


def sum(a, axis=None, dtype=None, out=None, keepdims=np._NoValue,
        initial=np._NoValue, where=np._NoValue):
    a = a if isinstance(a, np.ma.MaskedArray) else np.asarray(a)

    if axis == "column":
        if a.dtype.names is None:
            raise ValueError(f"'a' has no dtype.names with axis='column'")
        ret = np.sum(_along_column(sum, a), axis=0, dtype=dtype, out=out, keepdims=keepdims,
                     initial=initial, where=where)
    else:
        ret = np.sum(a, axis, dtype, out, keepdims, initial, where)
    if np.issubdtype(ret.dtype, np.object_):
        raise NotImplementedError("type of returned is object")
    return ret