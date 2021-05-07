import numpy as np


__all__ = ["is_numeric", "is_integer", "is_floating"]


def is_numeric(a):
    return is_floating(a) or is_integer(a)


def is_floating(a):
    return (
        isinstance(a, float) or
        (isinstance(a, np.ndarray) and np.issubdtype(a.dtype, np.floating))
    )


def is_integer(a):
    return (
        isinstance(a, int) or
        (isinstance(a, np.ndarray) and (
            np.issubdtype(a.dtype, np.signedinteger) or
            np.issubdtype(a.dtype, np.unsignedinteger)
        ))
    )
