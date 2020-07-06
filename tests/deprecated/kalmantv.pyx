from KalmanTV cimport KalmanTV as CKalmanTV

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
        n_meas (int): Size of the measure.
        n_state (int): Size of the state.
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
        mu_state (ndarray(n_state)): Transition_offsets defining the solution prior; denoted by :math:`c`.
        wgt_state (ndarray(n_state, n_state)): Transition matrix defining the solution prior; denoted by :math:`T`.
        var_state (ndarray(n_state, n_state)): Variance matrix defining the solution prior; denoted by :math:`R`.
        x_meas (ndarray(n_meas)): Measure at time n+1; denoted by :math:`y_{n+1}`.
        mu_meas (ndarray(n_meas)): Transition_offsets defining the measure prior; denoted by :math:`d`.
        wgt_meas (ndarray(n_meas, n_meas)): Transition matrix defining the measure prior; denoted by :math:`W`.
        var_meas (ndarray(n_meas, n_meas)): Variance matrix defining the measure prior; denoted by :math:`H`.

    """
    cdef CKalmanTV * ktv

    def __cinit__(self, int n_meas, int n_state):
        self.ktv = new CKalmanTV(n_meas, n_state)

    def __dealloc__(self):
        del self.ktv

    def predict(self,
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
        self.ktv.predict(& mu_state_pred[0], & var_state_pred[0, 0],
                          & mu_state_past[0], & var_state_past[0, 0],
                          & mu_state[0], & wgt_state[0, 0], & var_state[0, 0])
        return

    def update(self,
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
        self.ktv.update(& mu_state_filt[0], & var_state_filt[0, 0],
                        & mu_state_pred[0], & var_state_pred[0, 0],
                        & x_meas[0], & mu_meas[0],
                        & wgt_meas[0, 0], & var_meas[0, 0])
        return
    
    def filter(self,
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
        self.ktv.filter(& mu_state_pred[0], & var_state_pred[0, 0],
                        & mu_state_filt[0], & var_state_filt[0, 0],
                        & mu_state_past[0], & var_state_past[0, 0],
                        & mu_state[0], & wgt_state[0, 0], & var_state[0, 0],
                        & x_meas[0], & mu_meas[0],
                        & wgt_meas[0, 0], & var_meas[0, 0])
        return

    def smooth_mv(self,
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
        self.ktv.smooth_mv(& mu_state_smooth[0], & var_state_smooth[0, 0],
                           & mu_state_next[0], & var_state_next[0, 0],
                           & mu_state_filt[0], & var_state_filt[0, 0],
                           & mu_state_pred[0], & var_state_pred[0, 0],
                           & wgt_state[0, 0])
        return

    def smooth_sim(self,
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
        self.ktv.smooth_sim(& x_state_smooth[0], & x_state_next[0],
                            & mu_state_filt[0], & var_state_filt[0, 0],
                            & mu_state_pred[0], & var_state_pred[0, 0],
                            & wgt_state[0, 0], & z_state[0])
        return

    def smooth(self,
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
        self.ktv.smooth(& x_state_smooth[0], & mu_state_smooth[0],
                        & var_state_smooth[0, 0], & x_state_next[0],
                        & mu_state_next[0], & var_state_next[0, 0],
                        & mu_state_filt[0], & var_state_filt[0, 0],
                        & mu_state_pred[0], & var_state_pred[0, 0],
                        & wgt_state[0, 0], & z_state[0])
        return

    def state_sim(self,
                  double[::1] x_state,
                  double[::1] mu_state,
                  double[::1, :] var_state,
                  double[::1] z_state):
        """
        Simulates from a normal distribution with mean `mu_state`, variance `var_state`,
        and randomness `z_state` drawn from :math:`N(0, 1)`.
        """
        self.ktv.state_sim(& x_state[0], & mu_state[0],
                           & var_state[0, 0], & z_state[0])
        return