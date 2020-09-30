import numpy as np
import kalmantv.numba.scipy_linalg as scipy_linalg
import scipy as sp
import scipy.linalg
from numba import intc, float64
# from numba import jit
from numba.extending import register_jitable
from numba.experimental import jitclass

# --- helper functions ---------------------------------------------------------


@register_jitable
def _solveV(X, V, B, V_chol):
    r"""
    Solves X in :math:`VX = B`, where V is a variance matrix.

    Args:
        X (ndarray(N, M)): Returned matrix.
        V (ndarray(N, N)): Variance matrix.
        B (ndarray(N, M)): Second matrix.
        V_chol (ndarray(N, M)): Temporary matrix in which to store the cholesky factorization of V.

    Returns:
        (tuple):
        - **U** (ndarray(N, M)): Temp matrix.
        - **X** (ndarray(N, M)): X in :math:`VX = B`.

    """
    # no malloc in any of this
    V_chol[:] = V
    X[:] = B
    (V_chol, lower) = scipy.linalg.cho_factor(
        V_chol, overwrite_a=True, check_finite=False)
    X = scipy.linalg.cho_solve(
        (V_chol, lower), b=X, overwrite_b=True, check_finite=False)
    return


@register_jitable
def _quad_form(D, A, B, AB):
    r"""
    Calculates the quadratic form :math:`D += A B A'`.

    Args:
        D (ndarray(N, N)): Returned matrix.
        A (ndarray(N, M)): First matrix.
        B (ndarray(M, M)): Second matrix.
        AB (ndarray(N, M)): Temporary matrix in which to store `AB`.

    Returns:
        (tuple):
        - **D** (ndarray(N, N)): :math:`D = A B A' + D`.
        - **AB** (ndarray(N, M)): :math:`AB`.

    """
    AB[:] = np.dot(A, B)
    D += np.dot(AB, A.T)
    return


@register_jitable
def _mvn_sim(x, mu, V, z, V_chol):
    r"""
    Creates a draw from :math:`x \sim N(\mu, V)` from iid normals :math:`z \sim \N(0, I)`.

    Args:
        x (ndarray(n)): Generated draw.
        mu (ndarray(n)): Mean vector.
        V (ndarray(n, n)): Variance matrix.
        z (ndarray(n)): Random vector simulated from :math:`N(0, 1)`.
        V_chol (ndarray(n, n)): Temporary matrix to store lower Cholesky factor of `V`.

    Returns:
        (tuple):
        - **x** (ndarray(n)): Generated state.
        - **V_chol** (ndarray(n, n)): Lower Cholesky factor of `V`.
    """
    V_chol, lower = scipy.linalg.cho_factor(V, lower=True, check_finite=False)
    x[:] = z
    x = scipy_linalg.tri_mult((V_chol, lower), x,
                              overwrite_x=True, check_finite=False)
    x += mu
    return


_KalmanTV_spec = [
    ('n_meas', intc),
    ('n_state', intc),
    ('tmu_state', float64[::1]),
    ('tmu_state2', float64[::1]),
    ('tvar_state', float64[::1, :]),
    ('tvar_state2', float64[::1, :]),
    ('tvar_state3', float64[::1, :]),
    ('tmu_meas', float64[::1]),
    ('tvar_meas', float64[::1, :]),
    ('twgt_meas', float64[::1, :]),
    ('twgt_meas2', float64[::1, :]),
    ('llt_meas', float64[::1, :]),
    ('llt_state', float64[::1, :])
]


@ jitclass(_KalmanTV_spec)
class KalmanTV(object):
    r"""
    Create a Kalman Time-Varying object. The methods of the object can predict, update, sample and
    smooth the mean and variance of the Kalman Filter. This method is useful if one wants to track
    an object with streaming observations.

    The specific model we are using to track streaming observations is

    .. math::

        X_n = c_n + T_n X_{n-1} + R_n^{1/2} \epsilon_n

        y_n = d_n + W_n x_n + H_n^{1/2} \eta_n

    where :math:`\epsilon_n` and :math:`\eta_n` are independent :math:`N(0,1)` distributions and
    :math:`X_n` denotes the state of the Kalman Filter at time n and :math:`y_n` denotes the
    observation at time n.

    The variables of the model are defined below in the argument section. The methods of this class
    calculates :math:`\theta = (\mu, \Sigma)` for :math:`X_n` and the notation for
    the state at time n given observations from k is given by :math:`\theta_{n|K}`.

    Notes:
      - For best performance, all input arrrays should have contiguous memory in fortran-order.
      - Avoids memory allocation whenever possible.  One place this does not happen is in calculations of `A += B C`.  This is done with `A += np.dot(B, C)`, which involves malloc before the addition.

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
        mu_state (ndarray(n_state)): Transition offsets defining the solution prior; denoted by :math:`c_n`.
        wgt_state (ndarray(n_state, n_state)): Transition matrix defining the solution prior; denoted by :math:`T_n`.
        var_state (ndarray(n_state, n_state)): Variance matrix defining the solution prior; denoted by :math:`R_n`.
        x_meas (ndarray(n_meas)): Interrogated measure vector from `x_state`; :math:`y_n`.
        mu_meas (ndarray(n_meas)): Transition offsets defining the measure prior; denoted by :math:`d_n`.
        wgt_meas (ndarray(n_meas, n_meas)): Transition matrix defining the measure prior; denoted by :math:`W_n`.
        var_meas (ndarray(n_meas, n_meas)): Variance matrix defining the measure prior; denoted by :math:`H_n`.
        z_state (ndarray(n_state)): Random vector simulated from :math:`N(0, 1)`.

    """

    def __init__(self, n_meas: int, n_state: int) -> None:
        self.n_meas = n_meas
        self.n_state = n_state
        self.tmu_state = np.empty(n_state)
        self.tmu_state2 = np.empty(n_state)
        # note: Numba doesn't support `order='F'`, so emulate
        # via array transpose as documented here: https://numba.pydata.org/numba-doc/dev/user/faq.html#how-can-i-create-a-fortran-ordered-array
        self.tvar_state = np.empty((n_state, n_state)).T
        self.tvar_state2 = np.empty((n_state, n_state)).T
        self.tvar_state3 = np.empty((n_state, n_state)).T
        self.tmu_meas = np.empty(n_meas)
        self.tvar_meas = np.empty((n_meas, n_meas)).T
        self.twgt_meas = np.empty((n_state, n_meas)).T
        self.twgt_meas2 = np.empty((n_state, n_meas)).T
        self.llt_meas = np.empty((n_meas, n_meas)).T
        self.llt_state = np.empty((n_state, n_state)).T

    def predict(self,
                mu_state_pred,
                var_state_pred,
                mu_state_past,
                var_state_past,
                mu_state,
                wgt_state,
                var_state):
        r"""
        Perform one prediction step of the Kalman filter.
        Calculates :math:`\theta_{n|n-1}` from :math:`\theta_{n-1|n-1}`.

        Note: `mu_state_pred` and `mu_state_past` cannot refer to the same location in memory.  Same for `var_state_pred` and `var_state_past`.
        """
        mu_state_pred[:] = mu_state
        mu_state_pred += np.dot(wgt_state, mu_state_past)
        var_state_pred[:] = var_state
        _quad_form(var_state_pred, wgt_state, var_state_past, self.tvar_state)

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
        r"""
        Perform one update step of the Kalman filter.
        Calculates :math:`\theta_{n|n}` from :math:`\theta_{n|n-1}`.

        Note: `mu_state_filt` and `mu_state_pred` can refer to the same location in memory.  Same for `var_state_filt` and `var_state_pred`.
        """
        self.tmu_meas[:] = -mu_meas
        self.tmu_meas -= np.dot(wgt_meas, mu_state_pred)
        self.tvar_meas[:] = var_meas
        _quad_form(self.tvar_meas, wgt_meas, var_state_pred, self.twgt_meas)
        _solveV(self.twgt_meas2, self.tvar_meas, self.twgt_meas, self.llt_meas)
        self.tmu_meas += x_meas
        mu_state_filt[:] = mu_state_pred
        mu_state_filt += np.dot(self.twgt_meas2.T, self.tmu_meas)
        var_state_filt[:] = var_state_pred
        var_state_filt -= np.dot(self.twgt_meas2.T, self.twgt_meas)
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
        r"""
        Perform one step of the Kalman mean/variance smoother.
        Calculates :math:`\theta_{n|N}` from :math:`\theta_{n+1|N}`, :math:`\theta_{n|n}`, and :math:`\theta_{n+1|n}`.

        Note: `mu_state_smooth` and `mu_state_next` can refer to the same location in memory.  Same for `var_state_smooth` and `var_state_next`.
        """
        self.tvar_state[:] = np.dot(wgt_state, var_state_filt.T)
        _solveV(self.tvar_state, var_state_pred,
                self.tvar_state, self.llt_state)
        self.tmu_state[:] = mu_state_next - mu_state_pred
        self.tvar_state2[:] = var_state_next - var_state_pred
        self.tvar_state3[:] = np.dot(self.tvar_state.T, self.tvar_state2)
        mu_state_smooth[:] = mu_state_filt
        mu_state_smooth += np.dot(self.tvar_state.T, self.tmu_state)
        var_state_smooth[:] = var_state_filt
        var_state_smooth += np.dot(self.tvar_state3, self.tvar_state)
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
        r"""
        Perform one step of the Kalman sampling smoother.
        Calculates a draw :math:`x_{n|N}` from :math:`x_{n+1|N}`, :math:`\theta_{n|n}`, and :math:`\theta_{n+1|n}`.

        Note: `x_state_smooth` and `x_state_next` can refer to the same location in memory.
        """
        self.tvar_state[:] = np.dot(wgt_state, var_state_filt.T)
        _solveV(self.tvar_state2, var_state_pred,
                self.tvar_state, self.llt_state)
        self.tmu_state[:] = x_state_next - mu_state_pred
        self.tmu_state2[:] = mu_state_filt
        self.tmu_state2 += np.dot(self.tvar_state2.T, self.tmu_state)
        self.tvar_state3[:] = var_state_filt
        self.tvar_state3 -= np.dot(self.tvar_state2.T, self.tvar_state)
        _mvn_sim(x_state_smooth, self.tmu_state2, self.tvar_state3,
                 z_state, self.llt_state)
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
