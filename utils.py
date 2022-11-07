import jax.numpy as jnp
import numpy as np

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
