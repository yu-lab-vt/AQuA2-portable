"""Explicit MATLAB indexing; public Python coordinates are zero based."""

import numpy as np


def _indices(values):
    values = np.asarray(values)
    if values.dtype.kind not in "iuf":
        raise TypeError("Indices must be integers; implicit truncation is forbidden.")
    if values.dtype.kind == "f" and (not np.all(np.isfinite(values)) or np.any(values != np.trunc(values))):
        raise TypeError("MATLAB double indices must be finite, exact integers.")
    if np.any(values >= 2**63) or (values.dtype.kind != "u" and np.any(values < -(2**63))):
        raise IndexError("Indices must fit signed int64")
    return values.astype(np.int64)


def from_matlab_indices(values, size):
    values = _indices(values)
    if np.any(values < 1) or np.any(values > size):
        raise IndexError("MATLAB index outside [1, size].")
    return values.astype(np.int64) - 1


def to_matlab_indices(values, size):
    values = _indices(values)
    if np.any(values < 0) or np.any(values >= size):
        raise IndexError("Python index outside [0, size).")
    return values.astype(np.int64) + 1


def ind2sub(shape, matlab_indices):
    """MATLAB linear indices -> zero-based Python subscripts, column order."""
    indices = from_matlab_indices(matlab_indices, int(np.prod(shape)))
    return np.unravel_index(indices, shape, order="F")


def sub2ind(shape, *python_subscripts):
    """Zero-based Python subscripts -> MATLAB column-major linear indices."""
    subscripts = tuple(_indices(v) for v in python_subscripts)
    return np.ravel_multi_index(subscripts, shape, order="F") + 1
