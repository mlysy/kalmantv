cimport cython
import numpy as np
cimport numpy as np
from kalmantv.cython import KalmanTV

DTYPE = np.double
ctypedef np.double_t DTYPE_t

# FIXME: where should we be using memory views?  most likely in the inputs,
# but is there a risk of overwriting things?
# anywhere else?
# FIXME: need to replace all calls to np.linalg with low-level C calls.
# I can think of two ways to do this:
# 1.  Use BLAS/LAPACK numpy interface.  see `cython-linalg.ipynb`.
# 2.  Write a C++ class in Eigen to handle the remaining linalg operations.  For example:
#
#    ```
#    class LinalgUtils {
#    }
#    ```
#
#    The main issue here is that you should malloc everything at object instantiation time, so you'll need to specify the dimensions of all relevant operations.
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def kalman_ode_higher(fun, # FIXME: this should be cpdef'ed...right?
                      np.ndarray[DTYPE_t, ndim=1] x0_state,
                      double tmin,
                      double tmax,
                      int n_eval, 
                      np.ndarray[DTYPE_t, ndim=2] wgt_state,
                      np.ndarray[DTYPE_t, ndim=1] mu_state, 
                      np.ndarray[DTYPE_t, ndim=2] var_state,
                      np.ndarray[DTYPE_t, ndim=2] wgt_meas, 
                      np.ndarray[DTYPE_t, ndim=2] z_state_sim):
    # Dimensions of state and measure variables
    cdef int n_dim_meas = wgt_meas.shape[0]
    cdef int n_dim_state = len(mu_state)
    cdef int n_steps = n_eval + 1

    # argumgents for kalman_filter and kalman_smooth
    cdef np.ndarray[DTYPE_t, ndim=1] mu_meas = np.zeros(n_dim_meas,
                                                        dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, ndim=3] var_meass = np.zeros((n_dim_meas, n_dim_meas, n_steps),
                                                          dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, ndim=2] x_meass = np.zeros((n_dim_meas, n_steps),
                                                        dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, ndim=2] mu_state_filts = np.zeros((n_dim_state, n_steps),
                                                               dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, ndim=3] var_state_filts = np.zeros((n_dim_state, n_dim_state, n_steps),
                                                                dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, ndim=2] mu_state_preds = np.zeros((n_dim_state, n_steps),
                                                               dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, ndim=3] var_state_preds = np.zeros((n_dim_state, n_dim_state, n_steps),
                                                                dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, ndim=2] mu_state_smooths = np.zeros((n_dim_state, n_steps),
                                                                 dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, ndim=3] var_state_smooths = np.zeros((n_dim_state, n_dim_state, n_steps),
                                                                  dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, ndim=2] x_state_smooths = np.zeros((n_dim_state, n_steps),
                                                                dtype=DTYPE, order='F')

    # initialize things
    mu_state_filts[:, 0] = x0_state
    # FIXME: is np.dot cythonized?
    x_meass[:, 0] = x0_state.dot(wgt_meas.T)
    mu_state_preds[:, 0] = mu_state_filts[:, 0]
    var_state_preds[:, :, 0] = var_state_filts[:, :, 0]
    # forward pass
    ktv = KalmanTV(n_dim_meas, n_dim_state)
    # FIXME: type t
    for t in range(n_eval):
        # kalman filter:
        # 1. predict
        ktv.predict(mu_state_pred = mu_state_preds[:, t+1],
                    var_state_pred = var_state_preds[:, :, t+1],
                    mu_state_past = mu_state_filts[:, t],
                    var_state_past = var_state_filts[:, :, t],
                    mu_state = mu_state,
                    wgt_state = wgt_state,
                    var_state = var_state)
        # 2. chkrebtii interrogation
        var_meass[:, :, t+1] = np.linalg.multi_dot([wgt_meas, var_state_preds[:, :, t+1], wgt_meas.T]) 
        R_tt = np.linalg.cholesky(var_state_preds[:, :, t+1])
        x_state_tt = mu_state_preds[:, t+1] + R_tt.dot(z_state_sim[:, t]) 
        x_meass[:, t+1] = fun(x_state_tt, tmin + (tmax-tmin)*(t+1)/n_eval)
        # 3. update
        ktv.update(mu_state_filt = mu_state_filts[:, t+1],
                   var_state_filt = var_state_filts[:, :, t+1],
                   mu_state_pred = mu_state_preds[:, t+1],
                   var_state_pred = var_state_preds[:, :, t+1],
                   x_meas = x_meass[:, t+1],
                   mu_meas = mu_meas,
                   wgt_meas = wgt_meas,
                   var_meas = var_meass[:, :, t+1])

    # backward pass
    mu_state_smooths[:, n_eval] = mu_state_filts[:, n_eval]
    var_state_smooths[:, :, n_eval] = var_state_filts[:, :, n_eval]
    x_state_smooths[:, n_eval] = np.linalg.cholesky(var_state_smooths[:, :, n_eval]).dot(z_state_sim[:, 2*n_eval+1])
    x_state_smooths[:, n_eval] += mu_state_smooths[:, n_eval]
    # FIXME: type t?
    for t in reversed(range(n_eval)):
        ktv.smooth(x_state_smooth = x_state_smooths[:, t],
                   mu_state_smooth = mu_state_smooths[:, t],
                   var_state_smooth = var_state_smooths[:, :, t], 
                   x_state_next = x_state_smooths[:, t+1],
                   mu_state_next = mu_state_smooths[:, t+1],
                   var_state_next = var_state_smooths[:, :, t+1],
                   mu_state_filt = mu_state_filts[:, t],
                   var_state_filt = var_state_filts[:, :, t],
                   mu_state_pred = mu_state_preds[:, t+1],
                   var_state_pred = var_state_preds[:, :, t+1],
                   wgt_state = wgt_state,
                   z_state = z_state_sim[:, n_eval+t])
    
    return x_state_smooths, mu_state_smooths, var_state_smooths
