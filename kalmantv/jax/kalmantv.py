r"""
This module implements the Kalman Filter and Smoothing algorithms. The methods of the object can predict, update, sample and 
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

"""

import jax.numpy as jnp
import jax.scipy
import jax.scipy.linalg as jscl
from jax import jit
from jax.config import config
config.update("jax_enable_x64", True)

### Make sure to use Jax numpy instead of Numpy
@jit
def predict(mu_state_past,
            var_state_past,
            mu_state,
            wgt_state,
            var_state):
    r"""
    Perform one prediction step of the Kalman filter.
    Calculates :math:`\\theta_{n|n-1}` from :math:`\\theta_{n-1|n-1}`.

    Args:
        mu_state_past (ndarray(n_state)): Mean estimate for state at time n-1 given observations from 
            times [0...n-1]; :math:`\mu_{n-1|n-1}`. 
        var_state_past (ndarray(n_state, n_state)): Covariance of estimate for state at time n-1 given 
            observations from times [0...n-1]; :math:`\Sigma_{n-1|n-1}`.
        mu_state (ndarray(n_state)): Transition offsets defining the solution prior; denoted by :math:`\lambda`.
        wgt_state (ndarray(n_state, n_state)): Transition matrix defining the solution prior; denoted by :math:`Q`.
        var_state (ndarray(n_state, n_state)): Variance matrix defining the solution prior; denoted by :math:`R`.

    Returns:
        (tuple):
        - **mu_state_pred** (ndarray(n_steps, n_state)): Mean estimate for state at time n given observations from 
          times [0...n-1]; denoted by :math:`\mu_{n|n-1}`. 
        - **var_state_pred** (ndarray(n_steps, n_state)): Covariance of estimate for state at time n given 
          observations from times [0...n-1]; denoted by :math:`\Sigma_{n|n-1}`.

    """
    mu_state_pred = wgt_state.dot(mu_state_past - mu_state) + mu_state
    var_state_pred = jnp.linalg.multi_dot(
        [wgt_state, var_state_past, wgt_state.T]) + var_state
    return mu_state_pred, var_state_pred

@jit
def update(mu_state_pred,
           var_state_pred,
           x_meas,
           mu_meas,
           wgt_meas,
           var_meas):
    r"""
    Perform one update step of the Kalman filter.
    Calculates :math:`\\theta_{n|n}` from :math:`\\theta_{n|n-1}`.

    Args:
        mu_state_pred (ndarray(n_state)): Mean estimate for state at time n given observations from 
            times [0...n-1]; denoted by :math:`\mu_{n|n-1}`. 
        var_state_pred (ndarray(n_state, n_state)): Covariance of estimate for state at time n given 
            observations from times [0...n-1]; denoted by :math:`\Sigma_{n|n-1}`.
        x_meas (ndarray(n_meas)): Interrogated measure vector from `x_state`; :math:`y_n`.
        mu_meas (ndarray(n_meas)): Transition offsets defining the measure prior; denoted by :math:`d`.
        wgt_meas (ndarray(n_meas, n_state)): Transition matrix defining the measure prior; denoted by :math:`W`.
        var_meas (ndarray(n_meas, n_meas)): Variance matrix defining the measure prior; denoted by :math:`\Sigma_n`.
    
    Returns:
        (tuple):
        - **mu_state_filt** (ndarray(n_steps, n_state)): Mean estimate for state at time n given observations from 
          times [0...n]; denoted by :math:`\mu_{n|n}`. 
        - **var_state_filt** (ndarray(n_steps, n_state)): Covariance of estimate for state at time n given 
          observations from times [0...n]; denoted by :math:`\Sigma_{n|n}`.

    """
    mu_meas_pred = wgt_meas.dot(mu_state_pred) + mu_meas
    var_meas_state_pred = wgt_meas.dot(var_state_pred)
    var_meas_meas_pred = jnp.linalg.multi_dot(
        [wgt_meas, var_state_pred, wgt_meas.T]) + var_meas
    var_state_meas_pred = var_state_pred.dot(wgt_meas.T)
    var_state_temp = _solveV(var_meas_meas_pred, var_state_meas_pred.T).T
    mu_state_filt = mu_state_pred + \
        var_state_temp.dot(x_meas - mu_meas_pred)
    var_state_filt = var_state_pred - \
        var_state_temp.dot(var_meas_state_pred)
    return mu_state_filt, var_state_filt

@jit
def filter(mu_state_past,
           var_state_past,
           mu_state,
           wgt_state,
           var_state,
           x_meas,
           mu_meas,
           wgt_meas,
           var_meas):
    r"""
    Perform one step of the Kalman filter.
    Combines :func:`KalmanTV.predict` and :func:`KalmanTV.update` steps to get :math:`\\theta_{n|n}` from :math:`\\theta_{n-1|n-1}`.

    Args:
        mu_state_past (ndarray(n_state)): Mean estimate for state at time n-1 given observations from 
            times [0...n-1]; :math:`\mu_{n-1|n-1}`. 
        var_state_past (ndarray(n_state, n_state)): Covariance of estimate for state at time n-1 given 
            observations from times [0...n-1]; :math:`\Sigma_{n-1|n-1}`.
        mu_state (ndarray(n_state)): Transition offsets defining the solution prior; denoted by :math:`\lambda`.
        wgt_state (ndarray(n_state, n_state)): Transition matrix defining the solution prior; denoted by :math:`Q`.
        var_state (ndarray(n_state, n_state)): Variance matrix defining the solution prior; denoted by :math:`R`.
        x_meas (ndarray(n_meas)): Interrogated measure vector from `x_state`; :math:`y_n`.
        mu_meas (ndarray(n_meas)): Transition offsets defining the measure prior; denoted by :math:`d`.
        wgt_meas (ndarray(n_meas, n_state)): Transition matrix defining the measure prior; denoted by :math:`W`.
        var_meas (ndarray(n_meas, n_meas)): Variance matrix defining the measure prior; denoted by :math:`\Sigma_n`.

    Returns:
        (tuple):
        - **mu_state_pred** (ndarray(n_steps, n_state)): Mean estimate for state at time n given observations from 
          times [0...n-1]; denoted by :math:`\mu_{n|n-1}`. 
        - **var_state_pred** (ndarray(n_steps, n_state)): Covariance of estimate for state at time n given 
          observations from times [0...n-1]; denoted by :math:`\Sigma_{n|n-1}`.
        - **mu_state_filt** (ndarray(n_steps, n_state)): Mean estimate for state at time n given observations from 
          times [0...n]; denoted by :math:`\mu_{n|n}`. 
        - **var_state_filt** (ndarray(n_steps, n_state)): Covariance of estimate for state at time n given 
          observations from times [0...n]; denoted by :math:`\Sigma_{n|n}`.

    """
    mu_state_pred, var_state_pred = predict(mu_state_past=mu_state_past,
                                            var_state_past=var_state_past,
                                            mu_state=mu_state,
                                            wgt_state=wgt_state,
                                            var_state=var_state)
    mu_state_filt, var_state_filt = update(mu_state_pred=mu_state_pred,
                                           var_state_pred=var_state_pred,
                                           x_meas=x_meas,
                                           mu_meas=mu_meas,
                                           wgt_meas=wgt_meas,
                                           var_meas=var_meas)
    return mu_state_pred, var_state_pred, mu_state_filt, var_state_filt

@jit
def smooth_mv(mu_state_next,
              var_state_next,
              mu_state_filt,
              var_state_filt,
              mu_state_pred,
              var_state_pred,
              wgt_state):
    """
    Perform one step of the Kalman mean/variance smoother.
    Calculates :math:`\\theta_{n|N}` from :math:`\\theta_{n+1|N}`, :math:`\\theta_{n|n}`, and :math:`\\theta_{n+1|n}`.

    Args:
        mu_state_next (ndarray(n_state)): Mean estimate for state at time n+1 given observations from 
            times [0...N]; denoted by :math:`\mu_{n+1|N}`. 
        var_state_next (ndarray(n_state, n_state)): Covariance of estimate for state at time n+1 given 
            observations from times [0...N]; denoted by :math:`\Sigma_{n+1|N}`.
        mu_state_filt (ndarray(n_state)): Mean estimate for state at time n given observations from 
            times [0...n]; denoted by :math:`\mu_{n|n}`. 
        var_state_filt (ndarray(n_state, n_state)): Covariance of estimate for state at time n given 
            observations from times [0...n]; denoted by :math:`\Sigma_{n|n}`.
        mu_state_pred (ndarray(n_state)): Mean estimate for state at time n given observations from 
            times [0...n-1]; denoted by :math:`\mu_{n|n-1}`. 
        var_state_pred (ndarray(n_state, n_state)): Covariance of estimate for state at time n given 
            observations from times [0...n-1]; denoted by :math:`\Sigma_{n|n-1}`.
        wgt_state (ndarray(n_state, n_state)): Transition matrix defining the solution prior; denoted by :math:`Q`.
    
    Returns:
        (tuple):
        - **mu_state_smooth** (ndarray(n_state)): Mean estimate for state at time n given observations from 
          times [0...N]; denoted by :math:`\mu_{n|N}`.
        - **var_state_smooth** (ndarray(n_state, n_state)): Covariance of estimate for state at time n given 
          observations from times [0...N]; denoted by :math:`\Sigma_{n|N}`.

    """
    var_state_temp = var_state_filt.dot(wgt_state.T)
    var_state_temp_tilde = _solveV(var_state_pred, var_state_temp.T).T
    mu_state_smooth = mu_state_filt + \
        var_state_temp_tilde.dot(mu_state_next - mu_state_pred)
    var_state_smooth = var_state_filt + jnp.linalg.multi_dot(
        [var_state_temp_tilde, (var_state_next - var_state_pred), var_state_temp_tilde.T])
    return mu_state_smooth, var_state_smooth

@jit
def smooth_sim(x_state_next,
               mu_state_filt,
               var_state_filt,
               mu_state_pred,
               var_state_pred,
               wgt_state,
               z_state):
    r"""
    Perform one step of the Kalman sampling smoother.
    Calculates a draw :math:`x_{n|N}` from :math:`x_{n+1|N}`, :math:`\\theta_{n|n}`, and :math:`\\theta_{n+1|n}`.

    Args:
        x_state_next (ndarray(n_state)): Simulated state at time n+1 given observations from 
            times [0...N]; denoted by :math:`\x_{n+1|N}`. 
        mu_state_filt (ndarray(n_state)): Mean estimate for state at time n given observations from 
            times [0...n]; denoted by :math:`\mu_{n|n}`. 
        var_state_filt (ndarray(n_state, n_state)): Covariance of estimate for state at time n given 
            observations from times [0...n]; denoted by :math:`\Sigma_{n|n}`.
        mu_state_pred (ndarray(n_state)): Mean estimate for state at time n given observations from 
            times [0...n-1]; denoted by :math:`\mu_{n|n-1}`. 
        var_state_pred (ndarray(n_state, n_state)): Covariance of estimate for state at time n given 
            observations from times [0...n-1]; denoted by :math:`\Sigma_{n|n-1}`.
        wgt_state (ndarray(n_state, n_state)): Transition matrix defining the solution prior; denoted by :math:`Q`.
        z_state (ndarray(n_state)): Random vector simulated from :math:`N(0, 1)`.
    
    Returns:
        (ndarray(n_state)): Sample solution at time n given observations from times [0...N]; denoted by :math:`X_{n|N}`.

    """
    var_state_temp = var_state_filt.dot(wgt_state.T)
    var_state_temp_tilde = _solveV(var_state_pred, var_state_temp.T).T
    mu_state_sim = mu_state_filt + \
        var_state_temp_tilde.dot(x_state_next - mu_state_pred)
    var_state_sim = var_state_filt - \
        var_state_temp_tilde.dot(var_state_temp.T)
    x_state_smooth = _state_sim(mu_state_sim,
                                var_state_sim,
                                z_state)
    return x_state_smooth

@jit
def smooth(x_state_next,
           mu_state_next,
           var_state_next,
           mu_state_filt,
           var_state_filt,
           mu_state_pred,
           var_state_pred,
           wgt_state,
           z_state):
    r"""
    Perform one step of both Kalman mean/variance and sampling smoothers.
    Combines :func:`KalmanTV.smooth_mv` and :func:`KalmanTV.smooth_sim` steps to get :math:`x_{n|N}` and 
    :math:`\\theta_{n|N}` from :math:`\\theta_{n+1|N}`, :math:`\\theta_{n|n}`, and :math:`\\theta_{n+1|n}`.

    Args:
        x_state_next (ndarray(n_state)): Simulated state at time n+1 given observations from 
            times [0...N]; denoted by :math:`\x_{n+1|N}`. 
        mu_state_next (ndarray(n_state)): Mean estimate for state at time n+1 given observations from 
            times [0...N]; denoted by :math:`\mu_{n+1|N}`. 
        var_state_next (ndarray(n_state, n_state)): Covariance of estimate for state at time n+1 given 
            observations from times [0...N]; denoted by :math:`\Sigma_{n+1|N}`.
        mu_state_filt (ndarray(n_state)): Mean estimate for state at time n given observations from 
            times [0...n]; denoted by :math:`\mu_{n|n}`. 
        var_state_filt (ndarray(n_state, n_state)): Covariance of estimate for state at time n given 
            observations from times [0...n]; denoted by :math:`\Sigma_{n|n}`.
        mu_state_pred (ndarray(n_state)): Mean estimate for state at time n given observations from 
            times [0...n-1]; denoted by :math:`\mu_{n|n-1}`. 
        var_state_pred (ndarray(n_state, n_state)): Covariance of estimate for state at time n given 
            observations from times [0...n-1]; denoted by :math:`\Sigma_{n|n-1}`.
        wgt_state (ndarray(n_state, n_state)): Transition matrix defining the solution prior; denoted by :math:`Q`.
        z_state (ndarray(n_state)): Random vector simulated from :math:`N(0, 1)`.
    
    Returns:
        (tuple):
        - **x_state_smooth** (ndarray(n_state)): Sample solution at time n given observations from times [0...N]; 
          denoted by :math:`X_{n|N}`.
        - **mu_state_smooth** (ndarray(n_state)): Mean estimate for state at time n given observations from 
          times [0...N]; denoted by :math:`\mu_{n|N}`.
        - **var_state_smooth** (ndarray(n_state, n_state)): Covariance of estimate for state at time n given 
          observations from times [0...N]; denoted by :math:`\Sigma_{n|N}`.

    """
    mu_state_smooth, var_state_smooth = smooth_mv(mu_state_next=mu_state_next,
                                                  var_state_next=var_state_next,
                                                  mu_state_filt=mu_state_filt,
                                                  var_state_filt=var_state_filt,
                                                  mu_state_pred=mu_state_pred,
                                                  var_state_pred=var_state_pred,
                                                  wgt_state=wgt_state)
    x_state_smooth = smooth_sim(x_state_next=x_state_next,
                                mu_state_filt=mu_state_filt,
                                var_state_filt=var_state_filt,
                                mu_state_pred=mu_state_pred,
                                var_state_pred=var_state_pred,
                                wgt_state=wgt_state,
                                z_state=z_state)
    return  x_state_smooth, mu_state_smooth, var_state_smooth,


@jit
def _state_sim(mu_state,
               var_state,
               z_state):
    r"""
    Simulate X given its mean and variance.
    
    Args:
        mu_state (ndarray(n_state)): Mean vector.
        var_state (ndarray(n_state, n_state)): Variance matrix.
        z_state (ndarray(n_state)): Random vector simulated from :math:`N(0, 1)`.

    Returns:
        (ndarray(n_state)): Simulated vector.
    """
    x_state = jnp.linalg.cholesky(var_state)
    return jnp.dot(x_state, z_state) + mu_state

@jit
def _solveV(V, B):
    r"""
    Computes :math:`X = V^{-1}B` where V is a variance matrix.

    Args:
        V (ndarray(n_dim1, n_dim1)): Variance matrix V in :math:`X = V^{-1}B`.
        B (ndarray(n_dim1, n_dim2)): Matrix B in :math:`X = V^{-1}B`.

    Returns:
        (ndarray(n_dim1, n_dim2)): Matrix X in :math:`X = V^{-1}B`

    """
    L, low = jscl.cho_factor(V)
    return jscl.cho_solve((L, low), B)

@jit
def forecast(mu_state_pred,
             var_state_pred,
             mu_meas,
             wgt_meas,
             var_meas):
    r"""
    Forecasts the mean and variance of the measurement at time step n given observations from times [0...n-1].

    Args:
        mu_state_pred (ndarray(n_state)): Mean estimate for state at time n given observations from 
            times [0...n-1]; denoted by :math:`\mu_{n|n-1}`. 
        var_state_pred (ndarray(n_state, n_state)): Covariance of estimate for state at time n given 
            observations from times [0...n-1]; denoted by :math:`\Sigma_{n|n-1}`.
        mu_meas (ndarray(n_meas)): Transition offsets defining the measure prior; denoted by :math:`d`.
        wgt_meas (ndarray(n_meas, n_state)): Transition matrix defining the measure prior; denoted by :math:`W`.
        var_meas (ndarray(n_meas, n_meas)): Variance matrix defining the measure prior; denoted by :math:`\Sigma_n`.
    
    Returns:
        (tuple):
        - **mu_fore** (ndarray(n_meas)): Mean estimate for measurement at n given observations from [0...n-1]
        - **var_fore** (ndarray(n_meas, n_meas)): Covariance of estimate for state at time n given 
          observations from times [0...n-1]
    """
    mu_fore = wgt_meas.dot(mu_state_pred) + mu_meas
    var_fore = jnp.linalg.multi_dot(
        [wgt_meas, var_state_pred, wgt_meas.T]) + var_meas
    return mu_fore, var_fore
