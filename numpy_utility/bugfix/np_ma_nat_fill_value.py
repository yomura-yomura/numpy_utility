import numpy as np
import warnings


__all__ = ["is_not_fixed_yet", "fix"]


def is_not_fixed_yet():
    a = np.ma.empty(1, dtype=[
        ("1", [("2_1", "i8"), ("2_2", "M8[s]")], 14)
    ])
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error', "Upon accessing multidimensional field", UserWarning)
            _ = a["1"]
    except UserWarning:
        return True
    return False


def fix(a: np.ma.MaskedArray):
    _change_datetime_nat(a.fill_value)


def _change_datetime_nat(fill_fields, parent=None):
    if hasattr(fill_fields, "dtype"):
        if fill_fields.dtype.names is None:
            if np.issubdtype(fill_fields.dtype, np.datetime64):
                mask = np.isnat(fill_fields)
                if mask.any():
                    if fill_fields.ndim > 0:
                        fill_fields[mask] = 0
                    else:
                        parent[0][parent[1]] = 0
        else:
            # print(fill_fields.dtype.names)
            for n in fill_fields.dtype.names:
                _change_datetime_nat(fill_fields[n], parent=(fill_fields, n))
