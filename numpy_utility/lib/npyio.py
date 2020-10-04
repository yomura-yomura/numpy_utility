import numpy as np

__all__ = [
    "savez",
    "load"
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


@np.core.overrides.set_module("numpy_utility")
def load(file, *args, **kwargs):
    data = np.load(file, *args, **kwargs)

    masked_array_keys = [
        k[len(suffix_for_masked_array):]
        for k in data.keys()
        if k.startswith(suffix_for_masked_array)
    ]

    other_keys = [
        k
        for k in data.keys()
        if not k.startswith(suffix_for_masked_array)
        if k not in masked_array_keys
    ]

    # MaskedArray
    ret_data = {
        k: np.ma.MaskedArray(data[k], data[suffix_for_masked_array + k])
        for k in masked_array_keys
    }

    # Others
    ret_data.update({
        k: data[k]
        for k in other_keys
    })

    return ret_data

