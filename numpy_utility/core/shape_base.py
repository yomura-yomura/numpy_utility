import itertools
import numpy as np
from . import fromnumeric as _fromnumeric_module


__all__ = ["merge_arrays"]


def get_extended_dtype(type1, type2, *types):
    types = list(map(np.dtype, itertools.chain([type1, type2], types)))
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


def merge_arrays(arrays, validate_unique_columns=True):
    assert len(arrays) > 1
    common_dtype_names = list(set(arrays[0].dtype.names).intersection(*(a.dtype.names for a in arrays[1:])))

    common_dtypes = [
        (common_dtype_name, get_extended_dtype(*(a.dtype[common_dtype_name] for a in arrays)))
        for common_dtype_name in common_dtype_names
    ]

    assert len(common_dtypes) > 0
    # assert all(common_dtypes[0] == dtypes for dtypes in common_dtypes[1:])  # 全てのaが共通のdtypeを持つ
    # common_dtypes = common_dtypes[0]
    # common_dtype_names = [name for name, *_ in common_dtypes]
    arrays_dtypes = [
        [d for d in a.dtype.descr if d[0] not in common_dtype_names]
        for a in arrays
    ]
    arrays_dtype_names = [[name for name, *_ in dtype_descr] for dtype_descr in arrays_dtypes]
    other_dtypes = [dtype for arrays_dtype in arrays_dtypes for dtype in arrays_dtype]
    arrays_values_at_common_dtype = [np.unique(a[common_dtype_names].astype(common_dtypes, copy=False)) for a in arrays]

    if validate_unique_columns:
        if not all(shape_v == shape_a
                   for shape_v, shape_a in zip(map(np.shape, arrays_values_at_common_dtype), map(np.shape, arrays))):
            i = next(
                i
                for i, (shape_v, shape_a) in enumerate(
                    zip(map(np.shape, arrays_values_at_common_dtype), map(np.shape, arrays))
                )
                if shape_v != shape_a
            )
            raise ValueError(f"Non-unique values found in arrays[{i}][{common_dtype_names}]")

    values_at_common_dtype = np.unique([v for values in arrays_values_at_common_dtype for v in values])

    a = np.ma.empty(len(values_at_common_dtype), dtype=[*common_dtypes, *other_dtypes])
    a.mask = True

    for i, dtype_names in enumerate(arrays_dtype_names):
        indices = _fromnumeric_module.search_matched(values_at_common_dtype, arrays[i][common_dtype_names])
        for name in arrays[i].dtype.names:
            a[name][indices] = arrays[i][name]
    return a
