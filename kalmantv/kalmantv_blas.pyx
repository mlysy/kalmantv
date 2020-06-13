import numpy as np
cimport numpy as np

from kalmantv.blas_opt cimport *

DTYPE = np.double

cdef class KalmanTV:
    """
    Create a Kalman Time-Varying object. The methods of the object can predict, update, sample and 
    smooth the mean and variance of the Kalman Filter. This method is useful if one wants to track 
    an object with streaming observations.

    The specific model we are using to track streaming observations is

    .. math::

        X_n = c + T X_n-1 + R_n^{1/2} \epsilon_n

        y_n = d + W x_n + H_n^{1/2} \eta_n
    
    where :math:`\epsilon_n` and :math:`\eta_n` are independent :math:`N(0,1)` distributions and
    :math:`X_n` denotes the state of the Kalman Filter at time n and :math:`y_n` denotes the 
    observation at time n.

    The variables of the model are defined below in the argument section. The methods of this class
    calculates :math:`\\theta = (\mu, \Sigma)` for :math:`X_n` and the notation for
    the state at time n given observations from k is given by :math:`\\theta_{n|K}`.

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
        mu_state (ndarray(n_state)): Transition_offsets defining the solution prior; denoted by :math:`c_n`.
        wgt_state (ndarray(n_state, n_state)): Transition matrix defining the solution prior; denoted by :math:`T_n`.
        var_state (ndarray(n_state, n_state)): Variance matrix defining the solution prior; denoted by :math:`R_n`.
        x_meas (ndarray(n_meas)): Interrogated measure vector from `x_state`; :math:`y_n`.
        mu_meas (ndarray(n_meas)): Transition_offsets defining the measure prior; denoted by :math:`d_n`.
        wgt_meas (ndarray(n_meas, n_meas)): Transition matrix defining the measure prior; denoted by :math:`W_n`.
        var_meas (ndarray(n_meas, n_meas)): Variance matrix defining the measure prior; denoted by :math:`H_n`.
        z_state (ndarray(n_state)): Random vector simulated from :math:`N(0, 1)`.

    """
    """
    cdef int n_state, n_meas
    cdef double[::1] tmu_state
    cdef double[::1] tmu_state2
    cdef double[::1, :] tvar_state
    cdef double[::1, :] tvar_state2
    cdef double[::1, :] tvar_state3
    cdef double[::1] tmu_meas
    cdef double[::1, :] tvar_meas
    cdef double[::1, :] twgt_meas
    cdef double[::1, :] twgt_meas2
    cdef double[::1, :] llt_meas
    cdef double[::1, :] llt_state
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
        """
        Perform one prediction step of the Kalman filter.
        Calculates :math:`\\theta_{n|n-1}` from :math:`\\theta_{n-1|n-1}`.
        """
        cdef char* wgt_trans = 'N'
        cdef char* var_trans = 'N'
        cdef char* wgt_trans2 = 'T' 
        cdef int mu_alpha = 1, mu_beta = 1, var_alpha = 1, var_beta = 1
        vec_copy(mu_state, mu_state_pred)
        mat_vec_mult(wgt_trans, mu_alpha, wgt_state, mu_state_past, mu_beta, mu_state_pred)
        mat_copy(var_state, var_state_pred)
        mat_triple_mult(wgt_trans, var_trans, wgt_trans2, var_alpha, wgt_state, var_state_past,
                        self.tvar_state, wgt_state, var_beta, var_state_pred)
        
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
        """
        Perform one update step of the Kalman filter.
        Calculates :math:`\\theta_{n|n}` from :math:`\\theta_{n|n-1}`.
        """
        cdef char* wgt_trans = 'N'
        cdef char* wgt_trans2 = 'T'
        cdef char* var_trans = 'N'
        cdef int tmu_alpha = -1, tmu_beta = -1, tvar_alpha = 1, tvar_beta = 1
        cdef int tmu_alpha2 = 1
        cdef int mu_alpha = 1, mu_beta = 1, var_alpha = -1, var_beta = 1
        vec_copy(mu_meas, self.tmu_meas)
        mat_vec_mult(wgt_trans, tmu_alpha, wgt_meas, mu_state_pred, tmu_beta, self.tmu_meas)
        mat_copy(var_meas, self.tvar_meas)
        mat_triple_mult(wgt_trans, var_trans, wgt_trans2, tvar_alpha, wgt_meas, var_state_pred,
                        self.twgt_meas, wgt_meas, tvar_beta, self.tvar_meas)
        solveV(self.tvar_meas, self.twgt_meas, self.llt_meas, self.twgt_meas2)
        vec_add(tmu_alpha2, x_meas, self.tmu_meas)
        vec_copy(mu_state_pred, mu_state_filt)
        mat_vec_mult(wgt_trans2, mu_alpha, self.twgt_meas2, self.tmu_meas, mu_beta, mu_state_filt)
        mat_copy(var_state_pred, var_state_filt)
        mat_mult(wgt_trans2, wgt_trans, var_alpha, self.twgt_meas2, self.twgt_meas,
                 var_beta, var_state_filt)
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
        """
        Perform one step of the Kalman filter.
        Combines :func:`KalmanTV.predict` and :func:`KalmanTV.update` steps to get :math:`\\theta_{n|n}` from :math:`\\theta_{n-1|n-1}`.
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
        """
        Perform one step of the Kalman mean/variance smoother.
        Calculates :math:`\\theta_{n|N}` from :math:`\\theta_{n+1|N}`, :math:`\\theta_{n|n}`, and :math:`\\theta_{n+1|n}`.
        """
        cdef char* wgt_trans = 'N'
        cdef char* var_trans = 'T'
        cdef char* var_trans2 = 'N'
        cdef int tvar_alpha = 1, tvar_beta = 0, tmu_alpha = -1, tvar_alpha2 = -1, tvar_beta2 = 1
        cdef int mu_alpha = 1, mu_beta = 1, var_alpha = 1, var_beta = 1
        mat_mult(wgt_trans, var_trans, tvar_alpha, wgt_state, var_state_filt, tvar_beta,
                 self.tvar_state)
        solveV(var_state_pred, self.tvar_state, self.llt_state, self.tvar_state)
        vec_copy(mu_state_next, self.tmu_state)
        vec_add(tmu_alpha, mu_state_pred, self.tmu_state)
        mat_copy(var_state_next, self.tvar_state2)
        mat_add(tvar_alpha2, var_state_pred, tvar_beta2, self.tvar_state2)
        mat_mult(var_trans, var_trans2, tvar_alpha, self.tvar_state, self.tvar_state2, tvar_beta,
                 self.tvar_state3)
        vec_copy(mu_state_filt, mu_state_smooth)
        mat_vec_mult(var_trans, mu_alpha, self.tvar_state, self.tmu_state, mu_beta, mu_state_smooth)
        mat_copy(var_state_filt, var_state_smooth)
        mat_mult(var_trans2, var_trans2, var_alpha, self.tvar_state3, self.tvar_state, var_beta,
                 var_state_smooth)
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
        """
        Perform one step of the Kalman sampling smoother.
        Calculates a draw :math:`x_{n|N}` from :math:`x_{n+1|N}`, :math:`\\theta_{n|n}`, and :math:`\\theta_{n+1|n}`.
        """
        cdef char* wgt_trans = 'N'
        cdef char* var_trans = 'T'
        cdef char* var_trans2 = 'N'
        cdef int tvar_alpha = 1, tvar_beta = 0, tmu_alpha = -1
        cdef int tmu_alpha2 = 1, tmu_beta2 = 1, tvar_alpha2 = -1, tvar_beta2 = 1
        mat_mult(wgt_trans, var_trans, tvar_alpha, wgt_state, var_state_filt, tvar_beta,
                 self.tvar_state)
        solveV(var_state_pred, self.tvar_state, self.llt_state, self.tvar_state2)
        vec_copy(x_state_next, self.tmu_state)
        vec_add(tmu_alpha, mu_state_pred, self.tmu_state)
        vec_copy(mu_state_filt, self.tmu_state2)
        mat_vec_mult(var_trans, tmu_alpha2, self.tvar_state2, self.tmu_state, tmu_beta2, self.tmu_state2)
        mat_copy(var_state_filt, self.tvar_state3)
        mat_mult(var_trans, var_trans2, tvar_alpha2, self.tvar_state2, self.tvar_state, tvar_beta2,
                 self.tvar_state3)
        self.state_sim(x_state_smooth, self.tmu_state2,
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
        """
        Perform one step of both Kalman mean/variance and sampling smoothers.
        Combines :func:`KalmanTV.smooth_mv` and :func:`KalmanTV.smooth_sim` steps to get :math:`x_{n|N}` and 
        :math:`\\theta_{n|N}` from :math:`\\theta_{n+1|N}`, :math:`\\theta_{n|n}`, and :math:`\\theta_{n+1|n}`.
        """
        self.smooth_mv(mu_state_smooth, var_state_smooth,
                       mu_state_next, var_state_next,
                       mu_state_filt, var_state_filt,
                       mu_state_pred, var_state_pred, wgt_state)
        self.smooth_sim(x_state_smooth,x_state_next, 
                        mu_state_filt, var_state_filt,
                        mu_state_pred, var_state_pred, 
                        wgt_state, z_state)
        return
    
    cpdef void state_sim(self,
                         double[::1] x_state,
                         const double[::1] mu_state,
                         const double[::1, :] var_state,
                         const double[::1] z_state):
        """
        Simulates from a normal distribution with mean `mu_state`, variance `var_state`,
        and randomness `z_state` drawn from :math:`N(0, 1)`.
        """
        cdef char* trans = 'N'
        cdef char* uplo = 'L'
        cdef char* diag = 'N'
        cdef int x_alpha = 1
        chol_fact(var_state, self.llt_state)
        vec_copy(z_state, x_state)
        tri_vec_mult(uplo, trans, diag, self.llt_state, x_state)
        vec_add(x_alpha, mu_state, x_state)
        return
    