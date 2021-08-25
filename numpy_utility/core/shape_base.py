import itertools
import numpy as np
import collections
from . import fromnumeric as _fromnumeric_module


__all__ = ["merge_arrays"]


def get_extended_dtype(type1, type2, *types):
    types = list(map(np.dtype, itertools.chain([type1, type2], types)))
    assert all(len(type_.descr) == 1 for type_ in types)
    assert all(len(type_.descr[0]) == 2 for type_ in types)
    assert all(types[0].kind == type_.kind for type_ in types[1:])
    assert all(types[0].byteorder == type_.byteorder for type_ in types[1:])
    kind = types[0].kind
    byteorder = types[0].byteorder
    unit_itemsizes = {"i": 1, "f": 1, "S": 1, "U": 4}
    if kind in ("i", "f", "S", "U"):
        return f"{byteorder}{kind}{max(type_.itemsize for type_ in types) // unit_itemsizes[kind]}"
    elif kind == "M":
        raise NotImplementedError
    elif kind == "m":
        raise NotImplementedError
    else:
        raise TypeError(kind)


def take(a, idx):
    if isinstance(idx, tuple):
        if len(idx) == 0:
            return a
        else:
            return take(a[idx[0]], idx[1:])
    else:
        return a[idx]


def get_flatten_dtype_names(dtype: np.dtype):
    return [
        e
        for name in dtype.names
        for e in (
            [(name,)]
            if dtype[name].names is None else
            ((name, *dtypes) for dtypes in get_flatten_dtype_names(dtype[name]))
        )
    ]


def flatten_to_nested_dtype(flatten_dtypes: list):
    def f(common_dtype_1st_name, matched_dtypes):
        matched_dtypes = list(matched_dtypes)
        n_depths = [len(matched_dtype[0]) for matched_dtype in matched_dtypes]
        if len(n_depths) == 1 and n_depths[0] == 1:
            return common_dtype_1st_name, *matched_dtypes[0][1:]
        return (
            common_dtype_1st_name,
            flatten_to_nested_dtype([(names, dtype) for (_, *names), dtype in matched_dtypes])
        )
    return list(itertools.starmap(f, itertools.groupby(flatten_dtypes, lambda fd: fd[0][0])))


def merge_arrays(arrays, validate_unique_columns=True):
    assert len(arrays) > 1
    common_dtype_names = list(
        set(
            get_flatten_dtype_names(arrays[0].dtype)
        ).intersection(*(get_flatten_dtype_names(a.dtype) for a in arrays[1:]))
    )
    common_dtype_1st_names = [name_at_1dim for name_at_1dim, *_ in common_dtype_names]

    common_dtypes = flatten_to_nested_dtype([
        (common_dtype_name, get_extended_dtype(*(take(a.dtype, common_dtype_name) for a in arrays)))
        for common_dtype_name in common_dtype_names
    ])

    assert len(common_dtypes) > 0
    other_dtype_names = [
        [dtype_name for dtype_name in get_flatten_dtype_names(a.dtype) if dtype_name[0] not in common_dtype_1st_names]
        for a in arrays
    ]
    other_dtypes = flatten_to_nested_dtype([
        (dtype_name, *take(a.dtype, dtype_name).descr[0][1:])
        for a, dtype_names in zip(arrays, other_dtype_names)
        for dtype_name in dtype_names
    ])
    # other_dtypes = [dtype for arrays_dtype in other_dtypes for dtype in arrays_dtype]

    arrays_values_at_common_dtype = [
        np.unique(a[common_dtype_1st_names].astype(common_dtypes, copy=False))
        for a in arrays
    ]

    if validate_unique_columns:
        if not all(
            shape_v == shape_a
            for shape_v, shape_a in zip(map(np.shape, arrays_values_at_common_dtype), map(np.shape, arrays))
        ):
            i = next(
                i
                for i, (shape_v, shape_a) in enumerate(
                    zip(map(np.shape, arrays_values_at_common_dtype), map(np.shape, arrays))
                )
                if shape_v != shape_a
            )
            raise ValueError(f"Non-unique values found in arrays[{i}][{common_dtype_names}]")

    values_at_common_dtype = np.unique([
        v for values in arrays_values_at_common_dtype for v in values
    ]).astype(common_dtypes, copy=False)

    a = np.ma.empty(len(values_at_common_dtype), dtype=[*common_dtypes, *other_dtypes])
    a.mask = True

    for name in common_dtype_1st_names:
        a[name] = values_at_common_dtype[name]

    for i, dtype_names in enumerate(other_dtype_names):
        indices = _fromnumeric_module.search_matched(values_at_common_dtype, arrays[i][common_dtype_1st_names])
        for name in dtype_names:
            a_view = take(a, name)
            a_view[indices] = take(arrays[i], name)
    return a
