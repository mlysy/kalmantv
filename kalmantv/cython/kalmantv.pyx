from blas cimport *
import os
import numpy as np
cimport numpy as np


DTYPE = np.double

cpdef void state_sim(double[::1] x_state,
                     double[::1, :] llt_state,
                     const double[::1] mu_state,
                     const double[::1, :] var_state,
                     const double[::1] z_state):
    """
    Simulates from a normal distribution with mean `mu_state`, variance `var_state`,
    and randomness `z_state` drawn from :math:`N(0, 1)`.

    Args:
        x_state (ndarray(n_state)): Simulated state.
        llt_state (ndarray(n_state, n_state)): Temporary matrix to store cholesky factorization.
        mu_state (ndarray(n_state)): Mean vector.
        var_state (ndarray(n_state, n_state)): Variance matrix.
        z_state (ndarray(n_state)): Random vector simulated from :math:`N(0, 1)`.

    Returns:
        (tuple):
        - **x_state** (ndarray(n_state)): Simulated state.
        - **llt_state** (ndarray(n_state, n_state)): Temporary matrix to store cholesky factorization.

    """
    cdef char * trans = 'N'
    cdef char * uplo = 'L'
    cdef char * diag = 'N'
    cdef int x_alpha = 1
    chol_fact(llt_state, var_state)
    vec_copy(x_state, z_state)
    tri_vec_mult(x_state, uplo, trans, diag, llt_state)
    vec_add(x_state, x_alpha, mu_state)
    return

cdef class KalmanTV:
    r"""
    Create a Kalman Time-Varying object. The methods of the object can predict, update, sample and 
    smooth the mean and variance of the Kalman Filter. This method is useful if one wants to track 
    an object with streaming observations.

    The specific model we are using to approximate the solution :math:`x_n` is

    .. math::

        x_n = Q(x_{n-1} -\lambda) + \lambda + R_n^{1/2} \epsilon_n

        y_n = d + W x_n + \Sigma_n^{1/2} \eta_n

    where :math:`\epsilon_n` and :math:`\eta_n` are independent :math:`N(0,1)` distributions and
    :math:`y_n` denotes the model interrogation (observation) at time n.

    The variables of the model are defined below in the argument section. The methods of this class
    calculates :math:`\theta = (\mu, \Sigma)` for :math:`x_n` and the notation for
    the state at time n given observations from k is given by :math:`\theta_{n|K}`.

    Args:
        n_state (int): Number of state variables.
        n_meas (int): Number of measurement variables.
        mu_state_past (ndarray(n_state)): Mean estimate for state at time n-1 given observations from 
            times [0...n-1]; :math:`\mu_{n-1|n-1}`. 
        var_state_past (ndarray(n_state, n_state)): Covariance of estimate for state at time n-1 given 
            observations from times [0...n-1]; :math:`\Sigma_{n-1|n-1}`.
        mu_state_pred (ndarray(n_state)): Mean estimate for state at time n given observations from 
            times [0...n-1]; denoted by :math:`\mu_{n|n-1}`. 
        var_state_pred (ndarray(n_state, n_state)): Covariance of estimate for state at time n given 
            observations from times [0...n-1]; denoted by :math:`\Sigma_{n|n-1}`.
        mu_state_filt (ndarray(n_state)): Mean estimate for state at time n given observations from 
            times [0...n]; denoted by :math:`\mu_{n|n}`. 
        var_state_filt (ndarray(n_state, n_state)): Covariance of estimate for state at time n given 
            observations from times [0...n]; denoted by :math:`\Sigma_{n|n}`.
        mu_state_next (ndarray(n_state)): Mean estimate for state at time n+1 given observations from 
            times [0...N]; denoted by :math:`\mu_{n+1|N}`. 
        var_state_next (ndarray(n_state, n_state)): Covariance of estimate for state at time n+1 given 
            observations from times [0...N]; denoted by :math:`\Sigma_{n+1|N}`.
        x_state_smooths (ndarray(n_state)): Sample solution at time n given observations from times [0...N];
            denoted by :math:`X_{n|N}`
        mu_state_smooth (ndarray(n_state)): Mean estimate for state at time n given observations from 
            times [0...N]; denoted by :math:`\mu_{n|N}`.
        var_state_smooth (ndarray(n_state, n_state)): Covariance of estimate for state at time n given 
            observations from times [0...N]; denoted by :math:`\Sigma_{n|N}`.
        x_state (ndarray(n_state)): Simulated state vector; :math:`x_n`.
        mu_state (ndarray(n_state)): Transition offsets defining the solution prior; denoted by :math:`\lambda`.
        wgt_state (ndarray(n_state, n_state)): Transition matrix defining the solution prior; denoted by :math:`Q`.
        var_state (ndarray(n_state, n_state)): Variance matrix defining the solution prior; denoted by :math:`R`.
        x_meas (ndarray(n_meas)): Interrogated measure vector from `x_state`; :math:`y_n`.
        mu_meas (ndarray(n_meas)): Transition offsets defining the measure prior; denoted by :math:`d`.
        wgt_meas (ndarray(n_meas, n_state)): Transition matrix defining the measure prior; denoted by :math:`W`.
        var_meas (ndarray(n_meas, n_meas)): Variance matrix defining the measure prior; denoted by :math:`\Sigma_n`.
        z_state (ndarray(n_state)): Random vector simulated from :math:`N(0, 1)`.
        mu_fore (ndarray(n_meas)): Mean estimate for measurement at n given observations from [0...n-1]
        var_fore (ndarray(n_meas, n_meas)): Covariance of estimate for state at time n given 
            observations from times [0...n-1]
    """

    def __init__(self, int n_meas, int n_state):
        self.n_meas = n_meas
        self.n_state = n_state
        self.tmu_state = np.empty(n_state, dtype=DTYPE)
        self.tmu_state2 = np.empty(n_state, dtype=DTYPE)
        self.tvar_state = np.empty((n_state, n_state),
                                   dtype=DTYPE, order='F')
        self.tvar_state2 = np.empty((n_state, n_state),
                                    dtype=DTYPE, order='F')
        self.tvar_state3 = np.empty((n_state, n_state),
                                    dtype=DTYPE, order='F')
        self.tmu_meas = np.empty(n_meas, dtype=DTYPE)
        self.tvar_meas = np.empty((n_meas, n_meas),
                                  dtype=DTYPE, order='F')
        self.twgt_meas = np.empty((n_meas, n_state),
                                  dtype=DTYPE, order='F')
        self.twgt_meas2 = np.empty((n_meas, n_state),
                                   dtype=DTYPE, order='F')
        self.llt_meas = np.empty((n_meas, n_meas),
                                 dtype=DTYPE, order='F')
        self.llt_state = np.empty((n_state, n_state),
                                  dtype=DTYPE, order='F')

    cpdef void predict(self,
                       double[::1] mu_state_pred,
                       double[::1, :] var_state_pred,
                       const double[::1] mu_state_past,
                       const double[::1, :] var_state_past,
                       const double[::1] mu_state,
                       const double[::1, :] wgt_state,
                       const double[::1, :] var_state):
        r"""
        Perform one prediction step of the Kalman filter.
        Calculates :math:`\theta_{n|n-1}` from :math:`\theta_{n-1|n-1}`.
        """
        cdef char * wgt_trans = 'N'
        cdef char * var_trans = 'N'
        cdef char * wgt_trans2 = 'T'
        cdef int mu_alpha = 1, mu_beta = 1, var_alpha = 1, var_beta = 1
        vec_copy(mu_state_pred, mu_state)
        mat_vec_mult(mu_state_pred, wgt_trans, mu_alpha,
                     mu_beta, wgt_state, mu_state_past)
        mat_copy(var_state_pred, var_state)
        mat_triple_mult(var_state_pred, self.tvar_state, wgt_trans, var_trans, wgt_trans2,
                        var_alpha, var_beta, wgt_state, var_state_past, wgt_state)

        return

    cpdef void update(self,
                      double[::1] mu_state_filt,
                      double[::1, :] var_state_filt,
                      const double[::1] mu_state_pred,
                      const double[::1, :] var_state_pred,
                      const double[::1] x_meas,
                      const double[::1] mu_meas,
                      const double[::1, :] wgt_meas,
                      const double[::1, :] var_meas):
        r"""
        Perform one update step of the Kalman filter.
        Calculates :math:`\theta_{n|n}` from :math:`\theta_{n|n-1}`.
        """
        cdef char * wgt_trans = 'N'
        cdef char * wgt_trans2 = 'T'
        cdef char * var_trans = 'N'
        cdef int tmu_alpha = -1, tmu_beta = -1, tvar_alpha = 1, tvar_beta = 1
        cdef int tmu_alpha2 = 1
        cdef int mu_alpha = 1, mu_beta = 1, var_alpha = -1, var_beta = 1
        vec_copy(self.tmu_meas, mu_meas)
        mat_vec_mult(self.tmu_meas, wgt_trans, tmu_alpha,
                     tmu_beta, wgt_meas, mu_state_pred)
        mat_copy(self.tvar_meas, var_meas)
        mat_triple_mult(self.tvar_meas, self.twgt_meas, wgt_trans, var_trans, wgt_trans2,
                        tvar_alpha, tvar_beta, wgt_meas, var_state_pred, wgt_meas)
        solveV(self.llt_meas, self.twgt_meas2, self.tvar_meas, self.twgt_meas)
        vec_add(self.tmu_meas, tmu_alpha2, x_meas)
        vec_copy(mu_state_filt, mu_state_pred)
        mat_vec_mult(mu_state_filt, wgt_trans2, mu_alpha,
                     mu_beta, self.twgt_meas2, self.tmu_meas)
        mat_copy(var_state_filt, var_state_pred)
        mat_mult(var_state_filt, wgt_trans2, wgt_trans, var_alpha, var_beta,
                 self.twgt_meas2, self.twgt_meas)
        return

    cpdef void filter(self,
                      double[::1] mu_state_pred,
                      double[::1, :] var_state_pred,
                      double[::1] mu_state_filt,
                      double[::1, :] var_state_filt,
                      const double[::1] mu_state_past,
                      const double[::1, :] var_state_past,
                      const double[::1] mu_state,
                      const double[::1, :] wgt_state,
                      const double[::1, :] var_state,
                      const double[::1] x_meas,
                      const double[::1] mu_meas,
                      const double[::1, :] wgt_meas,
                      const double[::1, :] var_meas):
        r"""
        Perform one step of the Kalman filter.
        Combines :func:`KalmanTV.predict` and :func:`KalmanTV.update` steps to get :math:`\theta_{n|n}` from :math:`\theta_{n-1|n-1}`.
        """
        self.predict(mu_state_pred, var_state_pred,
                     mu_state_past, var_state_past,
                     mu_state, wgt_state, var_state)
        self.update(mu_state_filt, var_state_filt,
                    mu_state_pred, var_state_pred,
                    x_meas, mu_meas, wgt_meas, var_meas)
        return

    cpdef void smooth_mv(self,
                         double[::1] mu_state_smooth,
                         double[::1, :] var_state_smooth,
                         const double[::1] mu_state_next,
                         const double[::1, :] var_state_next,
                         const double[::1] mu_state_filt,
                         const double[::1, :] var_state_filt,
                         const double[::1] mu_state_pred,
                         const double[::1, :] var_state_pred,
                         const double[::1, :] wgt_state):
        r"""
        Perform one step of the Kalman mean/variance smoother.
        Calculates :math:`\theta_{n|N}` from :math:`\theta_{n+1|N}`, :math:`\theta_{n|n}`, and :math:`\theta_{n+1|n}`.
        """
        cdef char * wgt_trans = 'N'
        cdef char * var_trans = 'T'
        cdef char * var_trans2 = 'N'
        cdef int tvar_alpha = 1, tvar_beta = 0, tmu_alpha = -1, tvar_alpha2 = -1, tvar_beta2 = 1
        cdef int mu_alpha = 1, mu_beta = 1, var_alpha = 1, var_beta = 1
        mat_mult(self.tvar_state, wgt_trans, var_trans, tvar_alpha, tvar_beta, wgt_state,
                 var_state_filt)
        solveV(self.llt_state, self.tvar_state,
               var_state_pred, self.tvar_state)
        vec_copy(self.tmu_state, mu_state_next)
        vec_add(self.tmu_state, tmu_alpha, mu_state_pred)
        mat_copy(self.tvar_state2, var_state_next)
        mat_add(self.tvar_state2, tvar_alpha2, tvar_beta2, var_state_pred)
        mat_mult(self.tvar_state3, var_trans, var_trans2, tvar_alpha, tvar_beta, self.tvar_state,
                 self.tvar_state2)
        vec_copy(mu_state_smooth, mu_state_filt)
        mat_vec_mult(mu_state_smooth, var_trans, mu_alpha,
                     mu_beta, self.tvar_state, self.tmu_state)
        mat_copy(var_state_smooth, var_state_filt)
        mat_mult(var_state_smooth, var_trans2, var_trans2, var_alpha, var_beta, self.tvar_state3,
                 self.tvar_state)
        return

    cpdef void smooth_sim(self,
                          double[::1] x_state_smooth,
                          const double[::1] x_state_next,
                          const double[::1] mu_state_filt,
                          const double[::1, :] var_state_filt,
                          const double[::1] mu_state_pred,
                          const double[::1, :] var_state_pred,
                          const double[::1, :] wgt_state,
                          const double[::1] z_state):
        r"""
        Perform one step of the Kalman sampling smoother.
        Calculates a draw :math:`x_{n|N}` from :math:`x_{n+1|N}`, :math:`\theta_{n|n}`, and :math:`\theta_{n+1|n}`.
        """
        cdef char * wgt_trans = 'N'
        cdef char * var_trans = 'T'
        cdef char * var_trans2 = 'N'
        cdef int tvar_alpha = 1, tvar_beta = 0, tmu_alpha = -1
        cdef int tmu_alpha2 = 1, tmu_beta2 = 1, tvar_alpha2 = -1, tvar_beta2 = 1
        mat_mult(self.tvar_state, wgt_trans, var_trans, tvar_alpha, tvar_beta, wgt_state,
                 var_state_filt)
        solveV(self.llt_state, self.tvar_state2,
               var_state_pred, self.tvar_state)
        vec_copy(self.tmu_state, x_state_next)
        vec_add(self.tmu_state, tmu_alpha, mu_state_pred)
        vec_copy(self.tmu_state2, mu_state_filt)
        mat_vec_mult(self.tmu_state2, var_trans, tmu_alpha2,
                     tmu_beta2, self.tvar_state2, self.tmu_state)
        mat_copy(self.tvar_state3, var_state_filt)
        mat_mult(self.tvar_state3, var_trans, var_trans2, tvar_alpha2, tvar_beta2, self.tvar_state2,
                 self.tvar_state)
        state_sim(x_state_smooth, self.llt_state, self.tmu_state2,
                  self.tvar_state3, z_state)
        return

    cpdef void smooth(self,
                      double[::1] x_state_smooth,
                      double[::1] mu_state_smooth,
                      double[::1, :] var_state_smooth,
                      const double[::1] x_state_next,
                      const double[::1] mu_state_next,
                      const double[::1, :] var_state_next,
                      const double[::1] mu_state_filt,
                      const double[::1, :] var_state_filt,
                      const double[::1] mu_state_pred,
                      const double[::1, :] var_state_pred,
                      const double[::1, :] wgt_state,
                      const double[::1] z_state):
        r"""
        Perform one step of both Kalman mean/variance and sampling smoothers.
        Combines :func:`KalmanTV.smooth_mv` and :func:`KalmanTV.smooth_sim` steps to get :math:`x_{n|N}` and 
        :math:`\theta_{n|N}` from :math:`\theta_{n+1|N}`, :math:`\theta_{n|n}`, and :math:`\theta_{n+1|n}`.
        """
        self.smooth_mv(mu_state_smooth, var_state_smooth,
                       mu_state_next, var_state_next,
                       mu_state_filt, var_state_filt,
                       mu_state_pred, var_state_pred, wgt_state)
        self.smooth_sim(x_state_smooth, x_state_next,
                        mu_state_filt, var_state_filt,
                        mu_state_pred, var_state_pred,
                        wgt_state, z_state)
        return
    
    cpdef void forecast(self,
                        double[::1] mu_fore,
                        double[::1, :] var_fore,
                        const double[::1] mu_state_pred,
                        const double[::1, :] var_state_pred,
                        const double[::1] mu_meas,
                        const double[::1, :] wgt_meas,
                        const double[::1, :] var_meas):
        r"""
        Forecasts the mean and variance of the measurement at time step n given observations from times [0...n-1].
        """
        cdef char * wgt_trans = 'N'
        cdef char * wgt_trans2 = 'T'
        cdef char * var_trans = 'N'
        cdef int mu_alpha = 1, mu_beta = 1, var_alpha = 1, var_beta = 1
        vec_copy(mu_fore, mu_meas)
        mat_vec_mult(mu_fore, wgt_trans, mu_alpha,
                     mu_beta, wgt_meas, mu_state_pred)
        mat_copy(var_fore, var_meas)
        mat_triple_mult(var_fore, self.twgt_meas, wgt_trans, var_trans, wgt_trans2,
                        var_alpha, var_beta, wgt_meas, var_state_pred, wgt_meas)
        
        return
