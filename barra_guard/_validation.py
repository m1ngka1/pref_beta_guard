"""Small validation helpers shared by preparation and risk views."""

import numpy as np


def _labels(values, name):
    result = tuple(values)
    if (not result or any(not isinstance(x, str) or not x for x in result)
            or len(set(result)) != len(result)):
        raise ValueError(f'{name} must contain unique nonempty string labels')
    return result


def _array(value, shape, name):
    result = np.array(value, dtype=float, copy=True)
    if len(shape) == 1 and result.shape == (shape[0], 1):
        result = result[:, 0].copy()
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite with shape {shape}')
    result.flags.writeable = False
    return result
