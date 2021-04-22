"""
Time-Varying Kalman Filter and Smoother to track streaming observations.
"""
import numpy as np
import scipy as sp


class KalmanTV(object):
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
        wgt_meas (ndarray(n_meas, n_state)): Transition matrix defining the measure prior; denoted by :math:`W`.
        var_meas (ndarray(n_meas, n_meas)): Variance matrix defining the measure prior; denoted by :math:`H`.
        mu_fore (ndarray(n_meas)): Mean estimate for measurement at n given observations from [0...n-1]
        var_fore (ndarray(n_meas, n_meas)): Covariance of estimate for state at time n given 
            observations from times [0...n-1]
            
    """

    def __init__(self, n_meas, n_state):
        self._n_meas = n_meas
        self._n_state = n_state

    def predict(self,
                mu_state_pred,
                var_state_pred,
                mu_state_past,
                var_state_past,
                mu_state,
                wgt_state,
                var_state):
        """
        Perform one prediction step of the Kalman filter.
        Calculates :math:`\\theta_{n|n-1}` from :math:`\\theta_{n-1|n-1}`.
        """
        mu_state_pred[:] = wgt_state.dot(mu_state_past) + mu_state
        var_state_pred[:] = np.linalg.multi_dot(
            [wgt_state, var_state_past, wgt_state.T]) + var_state
        return

    def update(self,
               mu_state_filt,
               var_state_filt,
               mu_state_pred,
               var_state_pred,
               x_meas,
               mu_meas,
               wgt_meas,
               var_meas):
        """
        Perform one update step of the Kalman filter.
        Calculates :math:`\\theta_{n|n}` from :math:`\\theta_{n|n-1}`.
        """
        mu_meas_pred = wgt_meas.dot(mu_state_pred) + mu_meas
        var_meas_state_pred = wgt_meas.dot(var_state_pred)
        var_meas_meas_pred = np.linalg.multi_dot(
            [wgt_meas, var_state_pred, wgt_meas.T]) + var_meas
        var_state_meas_pred = var_state_pred.dot(wgt_meas.T)
        var_state_temp = np.linalg.solve(
            var_meas_meas_pred, var_state_meas_pred.T).T
        mu_state_filt[:] = mu_state_pred + \
            var_state_temp.dot(x_meas - mu_meas_pred)
        var_state_filt[:] = var_state_pred - \
            var_state_temp.dot(var_meas_state_pred)
        return

    def filter(self,
               mu_state_pred,
               var_state_pred,
               mu_state_filt,
               var_state_filt,
               mu_state_past,
               var_state_past,
               mu_state,
               wgt_state,
               var_state,
               x_meas,
               mu_meas,
               wgt_meas,
               var_meas):
        """
        Perform one step of the Kalman filter.
        Combines :func:`KalmanTV.predict` and :func:`KalmanTV.update` steps to get :math:`\\theta_{n|n}` from :math:`\\theta_{n-1|n-1}`.
        """
        self.predict(mu_state_pred=mu_state_pred,
                     var_state_pred=var_state_pred,
                     mu_state_past=mu_state_past,
                     var_state_past=var_state_past,
                     mu_state=mu_state,
                     wgt_state=wgt_state,
                     var_state=var_state)
        self.update(mu_state_filt=mu_state_filt,
                    var_state_filt=var_state_filt,
                    mu_state_pred=mu_state_pred,
                    var_state_pred=var_state_pred,
                    x_meas=x_meas,
                    mu_meas=mu_meas,
                    wgt_meas=wgt_meas,
                    var_meas=var_meas)
        return

    def smooth_mv(self,
                  mu_state_smooth,
                  var_state_smooth,
                  mu_state_next,
                  var_state_next,
                  mu_state_filt,
                  var_state_filt,
                  mu_state_pred,
                  var_state_pred,
                  wgt_state):
        """
        Perform one step of the Kalman mean/variance smoother.
        Calculates :math:`\\theta_{n|N}` from :math:`\\theta_{n+1|N}`, :math:`\\theta_{n|n}`, and :math:`\\theta_{n+1|n}`.
        """
        var_state_temp = var_state_filt.dot(wgt_state.T)
        var_state_temp_tilde = np.linalg.solve(
            var_state_pred, var_state_temp.T).T
        mu_state_smooth[:] = mu_state_filt + \
            var_state_temp_tilde.dot(mu_state_next - mu_state_pred)
        var_state_smooth[:] = var_state_filt + np.linalg.multi_dot(
            [var_state_temp_tilde, (var_state_next - var_state_pred), var_state_temp_tilde.T])
        return

    def smooth_sim(self,
                   x_state_smooth,
                   x_state_next,
                   mu_state_filt,
                   var_state_filt,
                   mu_state_pred,
                   var_state_pred,
                   wgt_state,
                   z_state):
        """
        Perform one step of the Kalman sampling smoother.
        Calculates a draw :math:`x_{n|N}` from :math:`x_{n+1|N}`, :math:`\\theta_{n|n}`, and :math:`\\theta_{n+1|n}`.
        """
        var_state_temp = var_state_filt.dot(wgt_state.T)
        var_state_temp_tilde = np.linalg.solve(
            var_state_pred, var_state_temp.T).T
        mu_state_sim = mu_state_filt + \
            var_state_temp_tilde.dot(x_state_next - mu_state_pred)
        var_state_sim = var_state_filt - \
            var_state_temp_tilde.dot(var_state_temp.T)
        # var_state_sim2 = var_state_sim.dot(var_state_sim.T)
        x_state_smooth[:] = np.linalg.cholesky(var_state_sim).dot(
            z_state)  # Use var_state_sim instead in the real algorithm
        x_state_smooth += mu_state_sim
        return

    def smooth(self,
               x_state_smooth,
               mu_state_smooth,
               var_state_smooth,
               x_state_next,
               mu_state_next,
               var_state_next,
               mu_state_filt,
               var_state_filt,
               mu_state_pred,
               var_state_pred,
               wgt_state,
               z_state):
        """
        Perform one step of both Kalman mean/variance and sampling smoothers.
        Combines :func:`KalmanTV.smooth_mv` and :func:`KalmanTV.smooth_sim` steps to get :math:`x_{n|N}` and 
        :math:`\\theta_{n|N}` from :math:`\\theta_{n+1|N}`, :math:`\\theta_{n|n}`, and :math:`\\theta_{n+1|n}`.
        """
        self.smooth_mv(mu_state_smooth=mu_state_smooth,
                       var_state_smooth=var_state_smooth,
                       mu_state_next=mu_state_next,
                       var_state_next=var_state_next,
                       mu_state_filt=mu_state_filt,
                       var_state_filt=var_state_filt,
                       mu_state_pred=mu_state_pred,
                       var_state_pred=var_state_pred,
                       wgt_state=wgt_state)
        self.smooth_sim(x_state_smooth=x_state_smooth,
                        x_state_next=x_state_next,
                        mu_state_filt=mu_state_filt,
                        var_state_filt=var_state_filt,
                        mu_state_pred=mu_state_pred,
                        var_state_pred=var_state_pred,
                        wgt_state=wgt_state,
                        z_state=z_state)
        return

    def state_sim(self,
                  x_state,
                  mu_state,
                  var_state,
                  z_state):
        x_state[:] = np.linalg.cholesky(var_state).dot(z_state)
        x_state += mu_state

    def forecast(self,
                 mu_fore,
                 var_fore,
                 mu_state_pred,
                 var_state_pred,
                 mu_meas,
                 wgt_meas,
                 var_meas):
        r"""
        Forecasts the mean and variance of the measurement at time step n given observations from times [0...n-1].
        """
        mu_fore[:] = wgt_meas.dot(mu_state_pred) + mu_meas
        var_fore[:] = np.linalg.multi_dot(
            [wgt_meas, var_state_pred, wgt_meas.T]) + var_meas
        return
