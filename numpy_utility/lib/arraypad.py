import numpy as np

__all__ = ["pad"]


def pad(array, pad_width, mode='constant', **kwargs):
    """
    arrayがMaskedArrayのときにも対応するように拡張。
    mode='constant'のときにkwargs['constant_values']が設定されていないと、padされた部分のmaskがTrueになる（値は0）。
    """
    if isinstance(array, np.ma.MaskedArray):
        if "constant_values" in kwargs:
            mask_kwargs = kwargs.copy()
            mask_kwargs["constant_values"] = False
        else:
            mask_kwargs = kwargs.copy()
            mask_kwargs["constant_values"] = True

        return np.ma.MaskedArray(
            pad(array.data, pad_width, mode, **kwargs),
            pad(array.mask, pad_width, mode, **mask_kwargs)
        )
    else:
        return np.pad(array, pad_width, mode, **kwargs)
