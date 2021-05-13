import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.linalg as jscl
import numpy as np

def _solveV(V, B):
    """
    Computes :math:`X = V^{-1}B` where V is a variance matrix.

    Args:
        V (ndarray(n_dim1, n_dim1)): Variance matrix V in :math:`X = V^{-1}B`.
        B (ndarray(n_dim1, n_dim2)): Matrix B in :math:`X = V^{-1}B`.

    Returns:
        (ndarray(n_dim1, n_dim2)): Matrix X in :math:`X = V^{-1}B`

    """
    L, low = jscl.cho_factor(V)
    return jscl.cho_solve((L, low), B)

def rand_vec(n, dtype=np.float64):
    """Generate a random vector."""
    x = jnp.array(np.random.randn(n))
    return x

def rand_mat(n, p=None, pd=True, dtype=np.float64, order='F'):
    """Generate a random matrix, positive definite if `pd = True`."""
    if p is None:
        p = n
    V = jnp.array(np.random.randn(n, p))
    if (p == n) & pd:
        V = jnp.matmul(V, V.T)
    return V.T


def rand_array(shape, dtype=np.float64, order='F'):
    """Generate a random array."""
    x = jnp.array(np.random.standard_normal(shape))
    return x.T
