import numpy as np
import pandas as pd


__all__ = ["to_numpy"]


def to_numpy(df: pd.DataFrame):
    dtype_dict = dict(df.dtypes)
    dtype_dict = {
        col: type_ if type_ != np.object_ else np.array(df[col].tolist()).dtype
        for col, type_ in dtype_dict.items()
    }
    return np.array(list(map(tuple, df.to_numpy())), dtype=np.dtype(list(dtype_dict.items())))
