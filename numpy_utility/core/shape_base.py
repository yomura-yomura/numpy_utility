import itertools
import numpy as np
from . import fromnumeric as _fromnumeric_module


__all__ = ["merge_arrays"]


def merge_arrays(arrays, validate_unique_columns=True):
    common_dtypes = [
        [s_d for f_d, s_d in itertools.product(f.descr, s.descr) if f_d == s_d]
        for f, s in itertools.combinations((a.dtype for a in arrays), 2)
    ]

    # assert len(arrays) == 2
    # assert len(common_dtypes) == 1
    assert all(common_dtypes[0] == dtypes for dtypes in common_dtypes[1:])  # 全てのaが共通のdtypeを持つ
    common_dtypes = common_dtypes[0]
    common_dtype_names = [name for name, *_ in common_dtypes]
    arrays_dtypes = [
        [d for d in a.dtype.descr if d not in common_dtypes]
        for a in arrays
    ]
    arrays_dtype_names = [[name for name, *_ in dtype_descr] for dtype_descr in arrays_dtypes]
    other_dtypes = [dtype for arrays_dtype in arrays_dtypes for dtype in arrays_dtype]
    arrays_values_at_common_dtype = [np.unique(a[common_dtype_names]) for a in arrays]

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
