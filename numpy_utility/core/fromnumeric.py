import numpy as np
import numpy.lib.recfunctions


__all__ = [
    "get_array_matched_with_boolean_array",
    "is_array",
    "combine_structured_arrays",
    "add_new_field_to",
    "change_field_format_to"
]


def get_array_matched_with_boolean_array(array, boolean_array, remove_all_masked_rows=False):
    assert(1 <= boolean_array.ndim <= 2)
    assert(array.size == boolean_array.shape[-1])

    new_array = np.ma.empty(boolean_array.shape, dtype=array.dtype)
    new_array[boolean_array] = array[np.where(boolean_array)[-1]]
    new_array.mask = ~boolean_array

    if remove_all_masked_rows:
        assert(new_array.ndim >= 2)
        new_array = new_array[~np.all(new_array.mask.view(bool), axis=-1)]

    return new_array


def is_array(obj):
    return isinstance(obj, (list, tuple, set, np.ndarray))


def combine_structured_arrays(a1, a2):
    assert a1.shape == a2.shape
    assert np.isin(a1.dtype.names, a2.dtype.names).any() == False
    new_fields = [(name, *sub) for name, *sub in a2.dtype.descr if name != ""]
    return add_new_field_to(a1, new_fields, a2)


def add_new_field_to(a, new_field, filled=None):
    assert is_array(new_field)
    if isinstance(new_field, tuple):
        new_field = [new_field]
    new_a = np.zeros(a.shape, a.dtype.descr + new_field)
    np.lib.recfunctions.recursive_fill_fields(a, new_a)
    if filled is not None:
        field_names = [name for name, *_ in new_field]
        new_a[field_names] = filled
    return new_a


def change_field_format_to(a, new_field_format):
    """
    new_field_format: dict type: {[field name]: [format]}
    """
    new_type = [(k, *sub) if k not in new_field_format.keys() else (k, new_field_format[k], *sub[1:])
                for k, *sub in a.dtype.descr]
    new_a = np.empty_like(a, new_type)
    np.lib.recfunctions.recursive_fill_fields(a, new_a)
    return new_a
