import numpy as np
import numpy.lib.recfunctions
from ..core import is_integer
import builtins
from collections.abc import Iterable


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
    "sum",
    "is_sorted",
    "to_tidy_data",
    # "fields_view"
]


def get_array_matched_with_boolean_array(a, boolean_array, remove_all_masked_rows=False):
    assert(1 <= boolean_array.ndim <= 2)
    # assert(a.size == boolean_array.shape[-1])
    a, boolean_array = np.broadcast_arrays(a, boolean_array)

    new_array = np.ma.empty(boolean_array.shape, dtype=a.dtype)
    # new_array[boolean_array] = a[np.where(boolean_array)[-1]]
    new_array[boolean_array] = a[boolean_array]
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
    return get_new_array_with_field_names(
        a, [field_name for field_name, *_ in a.dtype.descr if not np.isin(field_name, field).any()]
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
def from_dict(data, strict=True, use_common_shape=True):
    if isinstance(data, dict):
        new_array = [from_dict(v, strict, use_common_shape) for v in data.values()]
        masked_found = builtins.any(isinstance(na, np.ma.MaskedArray) for na in new_array)
    elif isinstance(data, np.ma.MaskedArray):
        return data
    else:
        return np.asarray(data)

    if use_common_shape:
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
            }, strict, use_common_shape)
            return flatten_ret.reshape(common_shape)
    else:
        common_shape = ()
        ndim_start = 0

    def get_dtype(a):
        pre_descr = a.dtype.descr[0][1:] if a.dtype.names is None else [a.dtype.descr]
        if a.ndim <= ndim_start:
            return pre_descr
        else:
            return (*pre_descr, a.shape[ndim_start:])

    new_dtype = [(k, *get_dtype(na)) for k, na in zip(data.keys(), new_array)]

    # if strict is True:
    #     shapes = [na.shape[:-len(common_shape)] for na in new_array]
    #     if not builtins.all(shapes[0] == shape for shape in shapes):
    #         raise ValueError(
    #             "\n".join([
    #                 f"Mismatch length:",
    #                 "\t Key, Shape",
    #                 *(f"\t {k}, {shape}" for k, shape in zip(data.keys(), shapes))
    #             ])
    #         )

    if masked_found:
        from .. import bugfix
        ret = np.ma.mrecords.fromarrays(
            new_array,
            dtype=new_dtype
        ).view(np.ma.MaskedArray)
        bugfix.np_ma_nat_fill_value.fix(ret)
        return ret
    else:
        # print(new_array)
        # return np.array(list(zip(*new_array)), new_dtype)
        # def to_tuple(a):
        #     return [
        #         tuple(to_tuple(ia)) if ia.ndim > 0 else ia
        #         for ia in a
        #     ]
        # return new_array, new_dtype
        # print(new_array)
        # print(new_dtype)
        # print(common_shape)
        if len(common_shape) == 0:
            return np.array([tuple(new_array)], new_dtype)
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
    def __init__(self, a, unique_keys, sort_by, iter_with=None):
        self.a = a
        self.sort_by = sort_by
        self.a_sorter = np.argsort(self.sort_by)

        unique_keys_sorter = np.argsort(unique_keys)
        self.sorted_unique_keys = unique_keys[unique_keys_sorter]
        self.unique_keys_unsorter = np.argsort(unique_keys_sorter)

        edges_matched_in_sorted_a = np.append(
            np.searchsorted(self.sort_by, self.sorted_unique_keys, sorter=self.a_sorter),
            a.size
        )
        self.left_edges_matched_in_sorted_a = edges_matched_in_sorted_a[:-1]
        self.right_edges_matched_in_sorted_a = edges_matched_in_sorted_a[1:]

        self.iter_with = iter_with
        if iter_with is not None:
            assert builtins.all(len(item) == len(a) for item in iter_with)

    def __iter__(self):
        # g = ((key, np.isin(self.field_names_to_sort_by, key)) for key in self.unique_keys)
        # assert is_sorted(self.edges_matched_in_sorted_a)
        g = (
            (
                self.sorted_unique_keys[i],
                self.a_sorter[self.left_edges_matched_in_sorted_a[i]:self.right_edges_matched_in_sorted_a[i]]
            )
            for i in self.unique_keys_unsorter
        )

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
        return len(self.sorted_unique_keys)


def groupby_given(a, by_given, iter_with=None, sort=False):
    assert is_array(a)
    assert is_array(by_given)

    if sort:
        unique_keys = np.unique(by_given, axis=0)
    else:
        unique_keys, indices = np.unique(by_given, return_index=True, axis=0)
        unique_keys = unique_keys[np.argsort(indices)]

    return GroupBy(a, unique_keys, by_given, iter_with)


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


def is_sorted(a):
    if isinstance(a, np.ma.MaskedArray):
        a = a.compressed()
    return builtins.all(i == i_order for i, i_order in enumerate(np.argsort(a)))


def to_tidy_data(dict_obj: dict, new_level_name="level_0", field_names=None):
    if field_names is not None and not is_array(field_names):
        field_names = [field_names]

    def to_ndarray(obj):
        if isinstance(obj, np.ndarray):
            return obj
        elif is_array(obj):
            from .. import ja
            ndim = ja.ndim(obj)
            if ndim == 1:
                obj = np.array(obj)
            elif ndim == 2:
                obj = np.array(obj, dtype="O")
                dtypes = [np.array(col.tolist()).dtype for col in np.rollaxis(obj, axis=1)]
                names = [f"f{i}" for i in range(len(dtypes))]
                obj = np.array(list(map(tuple, obj)), list(zip(names, dtypes)))
            else:
                raise NotImplementedError
        elif isinstance(obj, Iterable):
            obj = list(obj)
        else:
            raise TypeError(type(obj))
        return to_ndarray(obj)

    dict_obj = {k: to_ndarray(v) for k, v in dict_obj.items()}

    shapes = [v.shape[1:] for v in dict_obj.values()]
    assert builtins.all(shapes[0] == shape for shape in shapes[1:])
    value_shape = shapes[0]
    assert len(value_shape) in (0, 1)

    value_dtype_field_names_list = [v.dtype.names for v in dict_obj.values()]
    assert builtins.all(value_dtype_field_names_list[0] == field_name for field_name in value_dtype_field_names_list[1:])
    value_dtype_field_names = value_dtype_field_names_list[0]
    
    is_multi_columns = len(value_shape) != 0 or value_dtype_field_names is not None
    assert not (len(value_shape) != 0 and value_dtype_field_names is not None)

    if field_names is None:
        if is_multi_columns:
            n_var = value_shape[0] if len(value_shape) != 0 else len(value_dtype_field_names)
            field_names = tuple(str(i) for i in range(n_var))
        else:
            field_names = ("0",)

    key_dtype = np.array(list(dict_obj.keys())).dtype
    value_dtypes = [v.dtype for v in dict_obj.values()]

    def get_common_dtype(value_dtypes):
        if not builtins.all(value_dtypes[0].kind == v_dtype.kind for v_dtype in value_dtypes[1:]):
            raise TypeError("different dtypes found in dict_obj")
        value_dtype = max(value_dtypes, key=lambda dtype: dtype.itemsize)
        return value_dtype

    if value_dtype_field_names is not None:
        value_dtypes = [
            get_common_dtype([v_dtype[field_name] for v_dtype in value_dtypes])
            for field_name in value_dtype_field_names
        ]
        dtype = [(new_level_name, key_dtype), *((fn, v_dtype) for fn, v_dtype in zip(field_names, value_dtypes))]
    else:
        v_dtype = get_common_dtype(value_dtypes)
        dtype = [(new_level_name, key_dtype), *((fn, v_dtype) for fn in field_names)]

    if is_multi_columns:
        return np.array([(k, *iv) for k, v in dict_obj.items() for iv in v], dtype)
    else:
        return np.array([(k, iv) for k, v in dict_obj.items() for iv in v], dtype)


# def fields_view(arr, *field_names):
#     if np.ma.isMaskedArray(arr):
#         # mask and fill_value seem to be copies.
#         return np.ma.MaskedArray(
#             data=fields_view(arr.data, *field_names),
#             mask=fields_view(arr.mask, *field_names),
#             fill_value=fields_view(arr.fill_value, *field_names),
#             hard_mask=True
#         )
#
#     def get_dtype(dtype_fields, nested_field_names):
#         if is_array(nested_field_names):
#             if len(nested_field_names) == 0:
#                 raise ValueError("Unknown field_names format")
#             elif len(nested_field_names) == 1:
#                 return get_dtype(dtype_fields, nested_field_names[0])
#             else:
#                 next_dtype, *other_dtypes = nested_field_names
#                 dtype, offset = dtype_fields[next_dtype]
#                 next_dtype, next_offset = get_dtype(dtype.fields, other_dtypes)
#                 return next_dtype, offset + next_offset
#                 # return next_dtype, next_offset
#         else:
#             return dtype_fields[nested_field_names]
#
#     sep = "/"
#
#     dtype2 = np.dtype({
#         sep.join(nested_names): get_dtype(arr.dtype.fields, nested_names)
#         for nested_names in (e if is_array(e) else [e] for e in field_names)
#     })
#     print(dtype2)
#     return arr.getfield(dtype2)
#     # return np.ndarray(arr.shape, dtype2, arr, 0, arr.strides)
