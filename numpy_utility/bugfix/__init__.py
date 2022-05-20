import warnings
from . import np_ma_nat_fill_value

if np_ma_nat_fill_value.is_not_fixed_yet() is False:
    warnings.warn("A bug 'np_ma_nat_fill_values' has already been fixed.", DeprecationWarning)

__all__ = ["np_ma_nat_fill_value"]
