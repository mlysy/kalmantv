# helper functions for tests
import numpy as np
import itertools
import pandas as pd


def expand_grid(data_dict):
    """Create a dataframe from every combination of given values."""
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())


def rel_err(X1, X2):
    """Relative error between two numpy arrays."""
    return np.max(np.abs((X1.ravel() - X2.ravel())/X1.ravel()))


def rand_vec(n, dtype=np.float64):
    """Generate a random vector."""
    x = np.zeros(n, dtype=dtype)
    x[:] = np.random.randn(n)
    return x


def rand_mat(n, p=None, pd=True, dtype=np.float64, order='F'):
    """Generate a random matrix, positive definite if `pd = True`."""
    if p is None:
        p = n
    V = np.zeros((n, p), dtype=dtype, order=order)
    V[:] = np.random.randn(n, p)
    if (p == n) & pd:
        V[:] = np.matmul(V, V.T)
    return V


def rand_array(shape, dtype=np.float64, order='F'):
    """Generate a random array."""
    x = np.zeros(shape, dtype=dtype, order=order)
    x[:] = np.random.standard_normal(shape)
    return x
