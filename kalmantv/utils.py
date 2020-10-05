import os
import numpy as np
import scipy.linalg


def _solveV(V, B=None):
    """
    Computes :math:`X = V^{-1}B` where V is a variance matrix.

    Args:
        V (ndarray(n_dim1, n_dim1)): Variance matrix V in :math:`X = V^{-1}B`.
        B (ndarray(n_dim1, n_dim2)): Optional Matrix B in :math:`X = V^{-1}B`.  If `None` defaults to the `n_dim1 x n_dim1` identity matrix.

    Returns:
        (ndarray(n_dim1, n_dim2)): Matrix X in :math:`X = V^{-1}B`.

    """
    if B is None:
        B = np.identity(V.shape[0])
    L, low = scipy.linalg.cho_factor(V)
    return scipy.linalg.cho_solve((L, low), B)


def mvncond(mu, Sigma, icond):
    """
    Calculates A, b, and V such that :math:`y[!icond] | y[icond] \sim N(A y[icond] + b, V)`.

    Args:
        mu (ndarray(2*n_dim)): Mean of y.
        Sigma (ndarray(2*n_dim, 2*n_dim)): Covariance of y. 
        icond (ndarray(2*nd_dim)): Conditioning on the terms given.

    Returns:
        (tuple):
        - **A** (ndarray(n_dim, n_dim)): For :math:`y \sim N(\mu, \Sigma)` 
          such that :math:`y[!icond] | y[icond] \sim N(A y[icond] + b, V)` Calculate A.
        - **b** (ndarray(n_dim)): For :math:`y \sim N(\mu, \Sigma)` 
          such that :math:`y[!icond] | y[icond] \sim N(A y[icond] + b, V)` Calculate b.
        - **V** (ndarray(n_dim, n_dim)): For :math:`y \sim N(\mu, \Sigma)`
          such that :math:`y[!icond] | y[icond] \sim N(A y[icond] + b, V)` Calculate V.

    """
    # if y1 = y[~icond] and y2 = y[icond], should have A = Sigma12 * Sigma22^{-1}
    # A = np.dot(Sigma[np.ix_(~icond, icond)],
    #            sp.linalg.cho_solve(
    #                sp.linalg.cho_factor(Sigma[np.ix_(icond, icond)]),
    #                np.identity(sum(icond)))
    #            )
    Sigma12 = Sigma[np.ix_(~icond, icond)]
    A = np.dot(Sigma12, _solveV(Sigma[np.ix_(icond, icond)]))
    b = mu[~icond] - np.dot(A, mu[icond])  # mu1 - A * mu2
    V = Sigma[np.ix_(~icond, ~icond)] - \
        np.dot(A, Sigma12.T)  # Sigma11 - A * Sigma21
    return A, b, V


def get_include():
    r"""
    Return the directory that contains the kalmantv \\*.h header files.
    Extension modules that need to compile against kalmantv should use this
    function to locate the appropriate include directory.

    Copied from `numpy.get_include()`, except only works after kalmantv has been installed.
    """
    import kalmantv
    if numpy.show_config is None:
        # running from numpy source directory
        d = os.path.join(os.path.dirname(numpy.__file__), 'core', 'include')
    else:
        # using installed numpy core headers
        import numpy.core as core
        d = os.path.join(os.path.dirname(core.__file__), 'include')
    return d
