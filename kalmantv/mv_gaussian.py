import numpy as np


def ss2gss(wgt_state, mu_state, var_state, wgt_meas, mu_meas, var_meas):
    r"""
    Converts the state-space parameters:

    .. math::

        X_n = c_n + T_n X_{n-1} + R_n^{1/2} \epsilon_n

        y_n = d_n + W_n x_n + H_n^{1/2} \eta_n

    to the Gaussian process parameters with the model:

    .. math::

        Y_n = b_n + A_n Y_{n-1} + C_n \epsilon_n

    Args:
        wgt_state (ndarray(n_states, n_states, n_steps)): Transition matrices defining
            the solution prior; denoted by :math:`T_n`.
        mu_state (ndarray(n_state, n_steps)): Transition offsets defining the solution 
            prior; denoted by :math:`c_n`.
        var_state (ndarray(n_state, n_state, n_steps)): Variance matrices
            defining the solution prior; denoted by :math:`R_n`.
        wgt_meas (ndarray(n_meas, n_meas, n_steps)): Transition matrices defining the 
            measure prior; denoted by :math:`W_n`.
        mu_meas (ndarray(n_meas, n_steps)): Transition offsets defining the measure prior; 
            denoted by :math:`d_n`.
        var_meas (ndarray(n_meas, n_meas, n_steps)): Variance matrice defining 
            the measure prior; denoted by :math:`H_n`.

    Returns:
        (tuple):
        - **wgt_gss** (ndarray(n_gss, n_gss, steps)): Transition matrices in the Gaussian process; denoted by :math:`A_n`.
        - **mu_gss** (ndarray(n_gss, steps)): Transition offsets in the Gaussian process; denoted by :math:`b_n`.
        - **chol_gss** (ndarray(n_gss, n_gss, steps)): Cholesky of the variance matrices in the Gaussian process; denoted by :math:`C_n`.

    """
    # dimensions
    n_state, n_steps = mu_state.shape
    n_meas = mu_meas.shape[0]
    n_gss = n_state + n_meas

    # initialize gss variables
    wgt_gss = np.zeros((n_gss, n_gss, n_steps), order='F')
    mu_gss = np.zeros((n_gss, n_steps), order='F')
    chol_gss = np.zeros((n_gss, n_gss, n_steps), order='F')

    for i in range(n_steps):
        # convert var to chol
        chol_state = np.linalg.cholesky(var_state[:, :, i])
        chol_meas = np.linalg.cholesky(var_meas[:, :, i])

        # wgt_gss
        wgt = np.zeros((n_gss, n_gss), order='F')
        wgt[0:n_state, 0:n_state] = wgt_state[:, :, i]
        wgt[n_state:, 0:n_state] = wgt_meas[:, :, i].dot(wgt_state[:, :, i])
        wgt_gss[:, :, i] = wgt

        # mu_gss
        mu = np.zeros(n_gss)
        mu[0:n_state] = mu_state[:, i]
        mu[n_state:] = wgt_meas[:, :, i].dot(mu_state[:, i]) + mu_meas[:, i]
        mu_gss[:, i] = mu

        # chol_gss
        chol = np.zeros((n_gss, n_gss), order='F')
        chol[0:n_state, 0:n_state] = chol_state
        chol[n_state:, 0:n_state] = wgt_meas[:, :, i].dot(chol_state)
        chol[n_state:, n_state:] = chol_meas
        chol_gss[:, :, i] = chol

    wgt_gss[:, :, 0] = np.nan
    return wgt_gss, mu_gss, chol_gss


def mv_gaussian(wgt_gss, mu_gss, chol_gss):
    r"""
    Calculate the mean and variance of the joint density :math:`Y=(Y_0, Y_1, \ldots, Y_n)` where 
    :math:`Y_n` is the Gaussian process.

    Args:
        wgt_gss (ndarray(n_gss, n_gss, steps)): Transition matrices in the Gaussian process;
            denoted by :math:`A_n`.
        mu_gss (ndarray(n_gss, steps)): Transition offsets in the Gaussian process;
            denoted by :math:`b_n`.
        chol_gss (ndarray(n_gss, n_gss, steps)): Cholesky of the variance matrices in the 
            Gaussian process; denoted by :math:`C_n`.

    Returns:
        (tuple):
        - **gaussian_mu** (ndarray(n_gss, n_gss, steps)): Mean of the joint density.
        - **gaussian_var** (ndarray(n_gss, steps)): Variance of the joint density.

    """
    n_gss, n_steps = mu_gss.shape
    D = n_gss*n_steps
    An_m = np.zeros((n_gss, n_gss, n_steps, n_steps+1))
    for n in range(n_steps):
        for m in range(n_steps+1):
            if m > n:
                An_m[:, :, n, m] = np.eye(n_gss)
            elif n == m:
                An_m[:, :, n, m] = wgt_gss[:, :, n]
            else:
                diff = n-m
                wgt_diff = wgt_gss[:, :, m]
                for i in range(diff):
                    wgt_diff = np.matmul(wgt_gss[:, :, m+i+1], wgt_diff)
                An_m[:, :, n, m] = wgt_diff

    L = np.zeros((D, D))
    gaussian_mu = np.zeros(D)
    for n in range(n_steps):
        for m in range(n, n_steps):
            L[m*n_gss:(m+1)*n_gss, n*n_gss:(n+1) *
              n_gss] = np.matmul(An_m[:, :, m, n+1], chol_gss[:, :, n])
        for m in range(n+1):
            gaussian_mu[n*n_gss:(n+1)*n_gss] = gaussian_mu[n*n_gss:(n+1)
                                                           * n_gss] + An_m[:, :, n, m+1].dot(mu_gss[:, m])
    gaussian_var = L.dot(L.T)
    return gaussian_mu, gaussian_var
