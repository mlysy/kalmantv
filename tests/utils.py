# helper functions for tests
import numpy as np
import itertools
import pandas as pd


def expand_grid(data_dict):
    """Create a dataframe from every combination of given values."""
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())


def rel_err(X1, X2):
    """Relative error between two numpy arrays."""
    return np.max(np.abs((X1.ravel() - X2.ravel())/X1.ravel()))


def rand_vec(n, dtype=np.float64):
    """Generate a random vector."""
    x = np.zeros(n, dtype=dtype)
    x[:] = np.random.randn(n)
    return x


def rand_mat(n, p=None, pd=True, dtype=np.float64, order='F'):
    """Generate a random matrix, positive definite if `pd = True`."""
    if p is None:
        p = n
    V = np.zeros((n, p), dtype=dtype, order=order)
    V[:] = np.random.randn(n, p)
    if (p == n) & pd:
        V[:] = np.matmul(V, V.T)
    return V


def rand_array(shape, dtype=np.float64, order='F'):
    """Generate a random array."""
    x = np.zeros(shape, dtype=dtype, order=order)
    x[:] = np.random.standard_normal(shape)
    return x


def _test_predict(n_state, n_meas, ktv1, ktv2):
    mu_state_past = rand_vec(n_state)
    var_state_past = rand_mat(n_state)
    mu_state = rand_vec(n_state)
    wgt_state = rand_mat(n_state, pd=False)
    var_state = rand_mat(n_state)
    mu_state_pred1 = np.empty(n_state)
    var_state_pred1 = np.empty((n_state, n_state), order='F')
    mu_state_pred2 = np.empty(n_state)
    var_state_pred2 = np.empty((n_state, n_state), order='F')
    ktv1.predict(mu_state_pred1, var_state_pred1,
                 mu_state_past, var_state_past,
                 mu_state, wgt_state, var_state)
    ktv2.predict(mu_state_pred2, var_state_pred2,
                 mu_state_past, var_state_past,
                 mu_state, wgt_state, var_state)
    return mu_state_pred1, var_state_pred1, mu_state_pred2, var_state_pred2


def _test_update(n_state, n_meas, ktv1, ktv2):
    mu_state_pred = rand_vec(n_state)
    var_state_pred = rand_mat(n_state)
    x_meas = rand_vec(n_meas)
    mu_meas = rand_vec(n_meas)
    wgt_meas = rand_mat(n_meas, n_state, pd=False)
    var_meas = rand_mat(n_meas)
    mu_state_filt1 = np.empty(n_state)
    var_state_filt1 = np.empty((n_state, n_state), order='F')
    mu_state_filt2 = np.empty(n_state)
    var_state_filt2 = np.empty((n_state, n_state), order='F')
    ktv1.update(mu_state_filt1, var_state_filt1,
                mu_state_pred, var_state_pred,
                x_meas, mu_meas, wgt_meas, var_meas)
    ktv2.update(mu_state_filt2, var_state_filt2,
                mu_state_pred, var_state_pred,
                x_meas, mu_meas, wgt_meas, var_meas)
    return mu_state_filt1, var_state_filt1, mu_state_filt2, var_state_filt2


def _test_filter(n_state, n_meas, ktv1, ktv2):
    mu_state_past = rand_vec(n_state)
    var_state_past = rand_mat(n_state)
    mu_state = rand_vec(n_state)
    wgt_state = rand_mat(n_state, pd=False)
    var_state = rand_mat(n_state)
    x_meas = rand_vec(n_meas)
    mu_meas = rand_vec(n_meas)
    wgt_meas = rand_mat(n_meas, n_state, pd=False)
    var_meas = rand_mat(n_meas)
    mu_state_pred1 = np.empty(n_state)
    var_state_pred1 = np.empty((n_state, n_state), order='F')
    mu_state_filt1 = np.empty(n_state)
    var_state_filt1 = np.empty((n_state, n_state), order='F')
    mu_state_pred2 = np.empty(n_state)
    var_state_pred2 = np.empty((n_state, n_state), order='F')
    mu_state_filt2 = np.empty(n_state)
    var_state_filt2 = np.empty((n_state, n_state), order='F')
    ktv1.filter(mu_state_pred1, var_state_pred1,
                mu_state_filt1, var_state_filt1,
                mu_state_past, var_state_past,
                mu_state, wgt_state, var_state,
                x_meas, mu_meas, wgt_meas, var_meas)
    ktv2.filter(mu_state_pred2, var_state_pred2,
                mu_state_filt2, var_state_filt2,
                mu_state_past, var_state_past,
                mu_state, wgt_state, var_state,
                x_meas, mu_meas, wgt_meas, var_meas)
    return mu_state_pred1, var_state_pred1, mu_state_filt1, var_state_filt1, \
        mu_state_pred2, var_state_pred2, mu_state_filt2, var_state_filt2,


def _test_smooth_mv(n_state, n_meas, ktv1, ktv2):
    mu_state_next = rand_vec(n_state)
    var_state_next = rand_mat(n_state)
    mu_state_filt = rand_vec(n_state)
    var_state_filt = rand_mat(n_state)
    mu_state_pred = rand_vec(n_state)
    var_state_pred = rand_mat(n_state)
    wgt_state = rand_mat(n_state, pd=False)
    mu_state_smooth1 = np.empty(n_state)
    var_state_smooth1 = np.empty((n_state, n_state), order='F')
    mu_state_smooth2 = np.empty(n_state)
    var_state_smooth2 = np.empty((n_state, n_state), order='F')
    ktv1.smooth_mv(mu_state_smooth1, var_state_smooth1,
                   mu_state_next, var_state_next,
                   mu_state_filt, var_state_filt,
                   mu_state_pred, var_state_pred,
                   wgt_state)
    ktv2.smooth_mv(mu_state_smooth2, var_state_smooth2,
                   mu_state_next, var_state_next,
                   mu_state_filt, var_state_filt,
                   mu_state_pred, var_state_pred,
                   wgt_state)
    return mu_state_smooth1, var_state_smooth1, \
        mu_state_smooth2, var_state_smooth2


def _test_smooth_sim(n_state, n_meas, ktv1, ktv2):
    mu_state_past = rand_vec(n_state)
    var_state_past = rand_mat(n_state)
    x_state_next = rand_vec(n_state)
    mu_state = rand_vec(n_state)
    wgt_state = 0.01*rand_mat(n_state, pd=False)
    var_state = rand_mat(n_state)
    z_state = rand_vec(n_state)
    x_meas = rand_vec(n_meas)
    mu_meas = rand_vec(n_meas)
    wgt_meas = rand_mat(n_meas, n_state, pd=False)
    var_meas = rand_mat(n_meas)
    mu_state_pred1 = np.empty(n_state)
    var_state_pred1 = np.empty((n_state, n_state), order='F')
    mu_state_filt1 = np.empty(n_state)
    var_state_filt1 = np.empty((n_state, n_state), order='F')
    x_state_smooth1 = np.empty(n_state)
    mu_state_pred2 = np.empty(n_state)
    var_state_pred2 = np.empty((n_state, n_state), order='F')
    mu_state_filt2 = np.empty(n_state)
    var_state_filt2 = np.empty((n_state, n_state), order='F')
    x_state_smooth2 = np.empty(n_state)
    ktv1.filter(mu_state_pred1, var_state_pred1,
                mu_state_filt1, var_state_filt1,
                mu_state_past, var_state_past,
                mu_state, wgt_state, var_state,
                x_meas, mu_meas, wgt_meas, var_meas)
    ktv1.smooth_sim(x_state_smooth1, x_state_next,
                    mu_state_filt1, var_state_filt1,
                    mu_state_pred1, var_state_pred1,
                    wgt_state, z_state)
    ktv2.filter(mu_state_pred2, var_state_pred2,
                mu_state_filt2, var_state_filt2,
                mu_state_past, var_state_past,
                mu_state, wgt_state, var_state,
                x_meas, mu_meas, wgt_meas, var_meas)
    ktv2.smooth_sim(x_state_smooth2, x_state_next,
                    mu_state_filt2, var_state_filt2,
                    mu_state_pred2, var_state_pred2,
                    wgt_state, z_state)
    return x_state_smooth1, x_state_smooth2


def _test_smooth(n_state, n_meas, ktv1, ktv2):
    mu_state_past = rand_vec(n_state)
    var_state_past = rand_mat(n_state)
    x_state_next = rand_vec(n_state)
    mu_state_next = rand_vec(n_state)
    var_state_next = rand_mat(n_state)
    mu_state = rand_vec(n_state)
    wgt_state = 0.01*rand_mat(n_state, pd=False)
    var_state = rand_mat(n_state)
    z_state = rand_vec(n_state)
    x_meas = rand_vec(n_meas)
    mu_meas = rand_vec(n_meas)
    wgt_meas = rand_mat(n_meas, n_state, pd=False)
    var_meas = rand_mat(n_meas)
    mu_state_pred1 = np.empty(n_state)
    var_state_pred1 = np.empty((n_state, n_state), order='F')
    mu_state_filt1 = np.empty(n_state)
    var_state_filt1 = np.empty((n_state, n_state), order='F')
    x_state_smooth1 = np.empty(n_state)
    mu_state_smooth1 = np.empty(n_state)
    var_state_smooth1 = np.empty((n_state, n_state), order='F')
    mu_state_pred2 = np.empty(n_state)
    var_state_pred2 = np.empty((n_state, n_state), order='F')
    mu_state_filt2 = np.empty(n_state)
    var_state_filt2 = np.empty((n_state, n_state), order='F')
    x_state_smooth2 = np.empty(n_state)
    mu_state_smooth2 = np.empty(n_state)
    var_state_smooth2 = np.empty((n_state, n_state), order='F')
    ktv1.filter(mu_state_pred1, var_state_pred1,
                mu_state_filt1, var_state_filt1,
                mu_state_past, var_state_past,
                mu_state, wgt_state, var_state,
                x_meas, mu_meas, wgt_meas, var_meas)
    ktv1.smooth(x_state_smooth1, mu_state_smooth1,
                var_state_smooth1, x_state_next,
                mu_state_next, var_state_next,
                mu_state_filt1, var_state_filt1,
                mu_state_pred1, var_state_pred1,
                wgt_state, z_state)
    ktv2.filter(mu_state_pred2, var_state_pred2,
                mu_state_filt2, var_state_filt2,
                mu_state_past, var_state_past,
                mu_state, wgt_state, var_state,
                x_meas, mu_meas, wgt_meas, var_meas)
    ktv2.smooth(x_state_smooth2, mu_state_smooth2,
                var_state_smooth2, x_state_next,
                mu_state_next, var_state_next,
                mu_state_filt2, var_state_filt2,
                mu_state_pred2, var_state_pred2,
                wgt_state, z_state)
    return mu_state_smooth1, var_state_smooth1, x_state_smooth1, \
        mu_state_smooth2, var_state_smooth2, x_state_smooth2,

def _test_forecast(n_state, n_meas, ktv1, ktv2):
    mu_state_pred = rand_vec(n_state)
    var_state_pred = rand_mat(n_state)
    mu_meas = rand_vec(n_meas)
    wgt_meas = rand_mat(n_meas, n_state, pd=False)
    var_meas = rand_mat(n_meas)
    mu_fore1 = np.empty(n_meas)
    var_fore1 = np.empty((n_meas, n_meas), order='F')
    mu_fore2 = np.empty(n_meas)
    var_fore2 = np.empty((n_meas, n_meas), order='F')
    ktv1.forecast(mu_fore1, var_fore1,
                  mu_state_pred, var_state_pred,
                  mu_meas, wgt_meas, var_meas)
    ktv2.forecast(mu_fore2, var_fore2,
                  mu_state_pred, var_state_pred,
                  mu_meas, wgt_meas, var_meas)
    return mu_fore1, var_fore1, mu_fore2, var_fore2
