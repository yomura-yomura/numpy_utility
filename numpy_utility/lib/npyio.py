import numpy as np
from .. import ja
from .. import core


__all__ = [
    "savez",
    "load",
    "loadtxt"
]

suffix_for_masked_array = "_mask_"


@np.core.overrides.set_module("numpy_utility")
def savez(file, *args, **kwargs):
    """
    More generic savez:
        * Can save MaskedArray data with mask

    Note: Use numpy_utility.load to load MaskedArray data saved with this function.
    """

    kwargs_from_args = dict(
        (f"arr_{i}", arg)
        for i, arg in enumerate(args)
    )
    args = None

    if any(kwd in kwargs_from_args for kwd in kwargs):
        raise ValueError(
            "Cannot use un-named variables and keyword "
            ", ".join(kwd for kwd in kwargs if kwd in kwargs_from_args)
        )

    kwargs.update(kwargs_from_args)

    mask_dict = {
        suffix_for_masked_array+k: a.mask
        for k, a in kwargs.items()
        if isinstance(a, np.ma.MaskedArray)
    }
    if any(kwd in mask_dict.keys() for kwd in kwargs):
        raise ValueError(
            f"Cannot use the keywords starts with {suffix_for_masked_array}"
        )
    kwargs.update(mask_dict)

    np.savez(file, **kwargs)


class NpzFile:
    def __init__(self, file, mmap_mode, allow_pickle, fix_imports, encoding):
        self._data = np.load(file, mmap_mode, allow_pickle, fix_imports, encoding)

    def keys(self):
        keys = [k for k in self._data.keys() if not k.startswith(suffix_for_masked_array)]
        possible_mask_keys = [f"{suffix_for_masked_array}{k}" for k in keys]

        other_keys_start_with_suffix = [
            k for k in self._data.keys()
            if k.startswith(suffix_for_masked_array) and k not in possible_mask_keys
        ]
        if len(other_keys_start_with_suffix) > 0:
            raise NotImplementedError

        return tuple(keys)

    def values(self):
        return tuple(self[k] for k in self.keys())

    def __iter__(self):
        return self.keys()

    def __getitem__(self, key):
        a = self._data[key]
        mask_key = f"{suffix_for_masked_array}{key}"
        if mask_key in self._data.keys():
            return np.ma.MaskedArray(a, mask=self._data[mask_key])
        return a


@np.core.overrides.set_module("numpy_utility")
def load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII'):
    return NpzFile(file, mmap_mode, allow_pickle, fix_imports, encoding)
    # data = np.load(file, mmap_mode, allow_pickle, fix_imports, encoding)
    #
    # masked_array_keys = [
    #     k[len(suffix_for_masked_array):]
    #     for k in data.keys()
    #     if k.startswith(suffix_for_masked_array)
    # ]
    #
    # other_keys = [
    #     k
    #     for k in data.keys()
    #     if not k.startswith(suffix_for_masked_array)
    #     if k not in masked_array_keys
    # ]
    #
    # # MaskedArray
    # ret_data = {
    #     k: np.ma.MaskedArray(data[k], data[suffix_for_masked_array + k])
    #     for k in masked_array_keys
    # }
    #
    # # Others
    # ret_data.update({
    #     k: data[k]
    #     for k in other_keys
    # })
    #
    # return ret_data


def loadtxt(fname, dtype=float, comments='#', delimiter=None,
            converters=None, skiprows=0, usecols=None, unpack=False,
            ndmin=0, encoding='bytes', max_rows=None):
    """
    Can load a file with different number of columns.

    returns MaskedArray
    """

    assert converters is None
    # assert usecols is None
    assert unpack is False
    assert ndmin == 0
    assert encoding == 'bytes'

    if delimiter is not None:
        delimiter = delimiter.encode()

    comments = comments.encode()

    dtype = np.dtype(dtype)

    if usecols is None:
        def _line_split(line):
            return line.split(delimiter)
    else:
        def _line_split(line):
            return [e for i, e in enumerate(line.split(delimiter)) if i in usecols]

    with open(fname, "rb") as f:
        a = ja.from_jagged_array(
            [_line_split(line) for i, line in enumerate(f)
             if ((skiprows <= i) & ((max_rows is None) or (i <= max_rows)) &
                 (not line.startswith(comments)))],
            dtype=dtype if dtype.names is None else None
        )

    if dtype.names is not None:
        struct_a = np.ma.empty(a.shape[:-1], dtype)
        if len(dtype.names) != a.shape[-1]:
            raise ValueError("given dtype has different size of fields")

        for k, iter_a in zip(dtype.names, np.rollaxis(a, axis=-1)):
            struct_a[k][~iter_a.mask] = iter_a.compressed()
            struct_a[k].mask = iter_a.mask

        use_mask = np.any(core.any(struct_a, axis="column"))
        a = struct_a
    else:
        use_mask = np.any(a.mask)

    if use_mask:
        return a
    else:
        return a.data
