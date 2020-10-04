import numpy as np


__all__ = ["is_numeric", "is_integer", "is_floating"]


def is_numeric(a: np.ndarray):
    return is_floating(a) or is_integer(a)


def is_floating(a: np.ndarray):
    return np.issubdtype(a.dtype, np.floating)


def is_integer(a: np.ndarray):
    return np.issubdtype(a.dtype, np.signedinteger) or np.issubdtype(a.dtype, np.unsignedinteger)