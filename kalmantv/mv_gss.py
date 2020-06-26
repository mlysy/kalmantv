import numpy as np
import scipy.linalg as scl

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
    A = np.dot(Sigma[np.ix_(~icond, icond)], scl.cho_solve(
        scl.cho_factor(Sigma[np.ix_(icond, icond)]), np.identity(sum(icond))))
    b = mu[~icond] - np.dot(A, mu[icond])  # mu1 - A * mu2
    V = Sigma[np.ix_(~icond, ~icond)] - np.dot(A,
                                               Sigma[np.ix_(icond, ~icond)])  # Sigma11 - A * Sigma21
    return A, b, V

def ss2gss(wgt_states, mu_states, chol_states, wgt_meass, mu_meass, chol_meass):
    r"""
    Converts the state-space parameters:

    .. math::

        X_n = c_n + T_n X_{n-1} + R_n^{1/2} \epsilon_n

        y_n = d_n + W_n x_n + H_n^{1/2} \eta_n
    
    to the Gaussian process parameters with the model:

    .. math::

        Y_n = b_n + A_n Y_{n-1} + C_n \epsilon_n

    Args:
        wgt_states (ndarray(n_states, n_states, n_steps)): Transition matrices defining
            the solution prior; denoted by :math:`T_n`.
        mu_states (ndarray(n_state, n_steps)): Transition offsets defining the solution 
            prior; denoted by :math:`c_n`.
        chol_states (ndarray(n_state, n_state, n_steps)): Cholesky of the variance matrices
            defining the solution prior; denoted by :math:`R_n^{1/2}`.
        wgt_meass (ndarray(n_meas, n_meas, n_steps)): Transition matrices defining the 
            measure prior; denoted by :math:`W_n`.
        mu_meass (ndarray(n_meas, n_steps)): Transition offsets defining the measure prior; 
            denoted by :math:`d_n`.
        chol_meass (ndarray(n_meas, n_meas, n_steps)): Cholesky of the variance matrice defining 
            the measure prior; denoted by :math:`H_n^{1/2}`.
    
    Returns:
        (tuple):
        - **wgt_gsss** (ndarray(n_gss, n_gss, steps)): Transition matrices in the Gaussian process;
            denoted by :math:`A_n`.
        - **mu_gsss** (ndarray(n_gss, steps)): Transition offsets in the Gaussian process;
            denoted by :math:`b_n`.
        - **chol_gsss** (ndarray(n_gss, n_gss, steps)): Cholesky of the variance matrices in the 
            Gaussian process; denoted by :math:`C_n`.
            
    """
    # dimensions
    n_state, n_steps = mu_states.shape
    n_meas = mu_meass.shape[0]
    n_gss = n_state + n_meas
    
    # initialize gss variables
    wgt_gsss = np.zeros((n_gss, n_gss, n_steps), order='F')
    mu_gsss = np.zeros((n_gss, n_steps), order='F')
    chol_gsss = np.zeros((n_gss, n_gss, n_steps), order='F')
    
    for i in range(n_steps):
        # wgt_gss
        wgt_gss = np.zeros((n_gss, n_gss), order='F')
        wgt_gss[0:n_state, 0:n_state] = wgt_states[:, :, i]
        wgt_gss[n_state:, 0:n_state] = wgt_meass[:, :, i].dot(wgt_states[:, :, i])
        wgt_gsss[:, :, i] = wgt_gss
    
        # mu_gss
        mu_gss = np.zeros(n_gss)
        mu_gss[0:n_state] = mu_states[:, i]
        mu_gss[n_state:] = wgt_meass[:, :, i].dot(mu_states[:, i]) + mu_meass[:, i]
        mu_gsss[:, i] = mu_gss
        
        # chol_gss
        chol_gss = np.zeros((n_gss, n_gss), order='F')
        chol_gss[0:n_state, 0:n_state] = chol_states[:, :, i]
        chol_gss[n_state:, 0:n_state] = wgt_meass[:, :, i].dot(chol_states[:, :, i])
        chol_gss[n_state:, n_state:] = chol_meass[:, :, i]
        chol_gsss[:, :, i] = chol_gss
        
    wgt_gsss[:, :, 0] = np.nan
    return wgt_gsss, mu_gsss, chol_gsss

def mv_gss(wgt_gsss, mu_gsss, chol_gsss):
    r"""
    Calculate the mean and variance of the joint density :math:`Y=(Y_0, Y_1, \ldots, Y_n)` where 
    :math:`Y_n` is the Gaussian process.

    Args:
        wgt_gsss (ndarray(n_gss, n_gss, steps)): Transition matrices in the Gaussian process;
            denoted by :math:`A_n`.
        mu_gsss (ndarray(n_gss, steps)): Transition offsets in the Gaussian process;
            denoted by :math:`b_n`.
        chol_gsss (ndarray(n_gss, n_gss, steps)): Cholesky of the variance matrices in the 
            Gaussian process; denoted by :math:`C_n`.
    
    Returns:
        (tuple):
        - **gss_mean** (ndarray(n_gss, n_gss, steps)): Mean of the joint density.
        - **gss_var** (ndarray(n_gss, steps)): Variance of the joint density.
        
    """
    n_gss, n_steps = mu_gsss.shape
    D = n_gss*n_steps
    An_m = np.zeros((n_gss, n_gss, n_steps, n_steps+1))
    for n in range(n_steps):
        for m in range(n_steps+1):
            if m>n:
                An_m[:, :, n, m] = np.eye(n_gss)
            elif n==m:
                An_m[:, :, n, m] = wgt_gsss[:, :, n]
            else:
                diff = n-m
                wgt_diff = wgt_gsss[:, :, m]
                for i in range(diff):
                    wgt_diff = np.matmul(wgt_gsss[:, :, m+i+1], wgt_diff)
                An_m[:, :, n, m] = wgt_diff
    
    L = np.zeros((D, D))
    gss_mean = np.zeros(D)
    for n in range(n_steps):
        for m in range(n, n_steps):
            L[m*n_gss:(m+1)*n_gss, n*n_gss:(n+1)*n_gss] = np.matmul(An_m[:, :, m, n+1], chol_gsss[:, :, n])
        for m in range(n+1):
            gss_mean[n*n_gss:(n+1)*n_gss] = gss_mean[n*n_gss:(n+1)*n_gss] + An_m[:, :, n, m+1].dot(mu_gsss[:, m])
    gss_var = L.dot(L.T)
    return gss_mean, gss_var
