from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pickle

from typing import Any, List, Tuple

_M_SENTINEL = -99

# https://stackoverflow.com/questions/5347065/interweaving-two-numpy-arrays/5347492#5347492


def interweave(a: np.ndarray, b: np.ndarray) -> jnp.ndarray:
    """Interweave two arrays into a single array with alternating elements."""
    c = jnp.ravel(jnp.column_stack((a, b)))
    return c


def max_consecutive_trues(arrin: np.ndarray, tol: float = 1) -> Tuple[int, int]:
    """Return the start and end indices of the longest mostly-true window."""
    arr = np.array(arrin)
    l = len(arr)
    for i in range(l):
        for j in range(i):
            true_count = np.count_nonzero(arr[j:l - i + j])
            if true_count / (l - i) >= tol:
                start = j
                end = j + l - i
                return start, end
    return 0, 0


def sorti(li: List[List], i: int) -> List[List]:
    """Sort a list of lists by the i-th element in descending order."""
    li.sort(key=lambda x: x[i])
    return li[::-1]


def sign0(x: float) -> int:
    """Return the sign of x, treating zero as positive (+1)."""
    if x == 0:
        return 1
    else:
        return jnp.sign(x)


def get_retrofac(retro: bool) -> int:
    """Return -1 if retro is True, otherwise 1."""
    if retro:
        return -1
    else:
        return 1


def get_m(m: int) -> int:
    """Return 0 if m is the sentinel value, otherwise return m."""
    if m == _M_SENTINEL:
        return 0
    else:
        return m


def load_pickle_file(path: str) -> Any:
    """Load and return a Python object from a pickle file."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def all_close_to(array: np.ndarray, val: float) -> bool:
    """Check whether all elements of an array are close to a given value."""
    return np.allclose(array, np.ones_like(array)*val)


def linfunc(p: Tuple[float, float], x: np.ndarray) -> np.ndarray:
    m, c = p
    return m * x + c


def linfunc2(p: float, x: np.ndarray) -> np.ndarray:
    c = p
    return 2 * x + c
