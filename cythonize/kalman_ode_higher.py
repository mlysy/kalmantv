"""
.. module:: kalman_ode_higher

Probabilistic ODE solver based on the Kalman filter and smoother. 

"""
import numpy as np

from kalmantv.cython import KalmanTV

def kalman_ode_higher(fun, x0_state, tmin, tmax, n_eval, wgt_state, mu_state, var_state, wgt_meas, z_state_sim):
    """
    Probabilistic ODE solver based on the Kalman filter and smoother. Returns an approximate solution to the higher order ODE

    .. math:: w' x_t = F(x_t, t)
    
    on the time interval :math:`t \in [a, b]` with initial condition :math:`x_0 = x_0`. The corresponding variable names are

    The specific model we are using to approximate the solution :math:`x_n` is

    .. math::

        X_n = c + T X_{n-1} + R_n^{1/2} \epsilon_n

        y_n = d + W X_n + H_n^{1/2} \eta_n
    
    where :math:`\epsilon_n` and :math:`\eta_n` are independent :math:`N(0,1)` distributions and
    :math:`X_n = (x_n, y_n)` at time n and :math:`y_n` denotes the observation at time n.

    Args:
        fun (function): Higher order ODE function :math:`W x_t = F(x_t, t)` taking arguments :math:`x` and :math:`t`.
        x0_state (float): Initial value of the state variable :math:`x_t` at time :math:`t = 0`.
        tmin (int): First time point of the time interval to be evaluated; :math:`a`.
        tmax (int): Last time point of the time interval to be evaluated; :math:`b`.
        n_eval (int): Number of discretization points (:math:`N`) of the time interval that is evaluated, 
            such that discretization timestep is :math:`dt = b/N`.
        wgt_state (ndarray(n_dim_state, n_dim_state)): Transition matrix defining the solution prior; :math:`T`.
        mu_state (ndarray(n_dim_state)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_dim_state, n_dim_state)): Variance matrix defining the solution prior; :math:`R`.
        wgt_meas (ndarray(n_dim_state)): Transition matrix defining the measure prior; :math:`W`.
        
    Returns:
        (tuple):
        - **x_state_smooths** (ndarray(n_timesteps, n_dim_state)): Sample solution at time t given observations from times [0...N] for
          :math:`t = 0,1/N,\ldots,1`.
        - **mu_state_smooths** (ndarray(n_timesteps, n_dim_state)): Posterior mean of the solution process :math:`y_n` at times 
          :math:`t = 0,1/N,\ldots,1`.
        - **var_state_smooths** (ndarray(n_timesteps, n_dim_state, n_dim_state)): Posterior variance of the solution process at
          times :math:`t = 0,1/N,\ldots,1`.

    """
    # Dimensions of state and measure variables
    n_dim_meas = wgt_meas.shape[0]
    n_dim_state = len(mu_state)
    n_timesteps = n_eval + 1

    # argumgents for kalman_filter and kalman_smooth
    mu_meas = np.zeros(n_dim_meas)
    var_meass = np.zeros((n_timesteps, n_dim_meas, n_dim_meas))
    x_meass = np.zeros((n_timesteps,n_dim_meas))
    mu_state_filts = np.zeros((n_timesteps, n_dim_state))
    var_state_filts = np.zeros((n_timesteps, n_dim_state, n_dim_state))
    mu_state_preds = np.zeros((n_timesteps, n_dim_state))
    var_state_preds = np.zeros((n_timesteps, n_dim_state, n_dim_state))
    mu_state_smooths = np.zeros((n_timesteps, n_dim_state))
    var_state_smooths = np.zeros((n_timesteps, n_dim_state, n_dim_state))
    x_state_smooths = np.zeros((n_timesteps, n_dim_state))

    # initialize things
    mu_state_filts[0] = x0_state
    x_meass[0] = x0_state.dot(wgt_meas.T)
    mu_state_preds[0] = mu_state_filts[0]
    var_state_preds[0] = var_state_filts[0]

    # forward pass
    ktv = KalmanTV(n_dim_meas, n_dim_state)
    for t in range(n_eval):
        ktv.predict(mu_state_pred = mu_state_preds[t+1],
                    var_state_pred = var_state_preds[t+1],
                    mu_state_past = mu_state_filts[t],
                    var_state_past = var_state_filts[t],
                    mu_state = mu_state,
                    wgt_state = wgt_state,
                    var_state = var_state)

        var_meass[t+1] = np.linalg.multi_dot([wgt_meas, var_state_preds[t+1], wgt_meas.T]) 
        I_tt = np.random.normal(loc=0.0, scale=1.0, size=n_dim_state)
        R_tt = np.linalg.cholesky(var_state_preds[t+1])
        x_state_tt = mu_state_preds[t+1] + R_tt.dot(I_tt) 
        x_meass[t+1] = fun(x_state_tt, tmin + (tmax-tmin)*(t+1)/n_eval)

        ktv.update(mu_state_filt = mu_state_filts[t+1],
                   var_state_filt = var_state_filts[t+1],
                   mu_state_pred = mu_state_preds[t+1],
                   var_state_pred = var_state_preds[t+1],
                   x_meas = x_meass[t+1],
                   mu_meas = mu_meas,
                   wgt_meas = wgt_meas,
                   var_meas = var_meass[t+1])

    # backward pass
    mu_state_smooths[-1] = mu_state_filts[-1]
    var_state_smooths[-1] = var_state_filts[-1]
    x_state_smooths[-1] = np.linalg.cholesky(var_state_smooths[-1]).dot(z_state_sim[-1])
    x_state_smooths[-1] += mu_state_smooths[-1]
    #x_state_smooths[-1] = np.random.multivariate_normal(mu_state_smooths[-1], var_state_smooths[-1], tol=1e-6)
    for t in reversed(range(n_eval)):
        ktv.smooth(x_state_smooth = x_state_smooths[t],
                   mu_state_smooth = mu_state_smooths[t],
                   var_state_smooth = var_state_smooths[t], 
                   x_state_next = x_state_smooths[t+1],
                   mu_state_next = mu_state_smooths[t+1],
                   var_state_next = var_state_smooths[t+1],
                   mu_state_filt = mu_state_filts[t],
                   var_state_filt = var_state_filts[t],
                   mu_state_pred = mu_state_preds[t+1],
                   var_state_pred = var_state_preds[t+1],
                   wgt_state = wgt_state,
                   z_state = z_state_sim[t])
    
    return x_state_smooths, mu_state_smooths, var_state_smooths
