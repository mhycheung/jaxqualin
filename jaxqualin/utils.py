import jax.numpy as jnp
import numpy as np
import pickle

# https://stackoverflow.com/questions/5347065/interweaving-two-numpy-arrays/5347492#5347492


def interweave(a, b):
    c = jnp.ravel(jnp.column_stack((a, b)))
    return c


def max_consecutive_trues(arrin, tol=1):
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


def sorti(li, i):
    li.sort(key=lambda x: x[i])
    return li[::-1]


def npsign0(x):
    if x == 0:
        return 1
    else:
        return np.sign(x)


def jnpsign0(x):
    if x == 0:
        return 1
    else:
        return jnp.sign(x)


def get_retrofac(retro):
    if retro:
        return -1
    else:
        return 1


def get_m(m):
    if m == -99:
        return 0
    else:
        return m


def load_pickle_file(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data
