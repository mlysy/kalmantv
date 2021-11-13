import unittest
import numpy as np
from kalmantv_py import KalmanTV as KalmanTV_py
from kalmantv_jax import *
import sys
sys.path.append("..")
from utils import *

def rel_err(X1, X2):
    """Relative error between two numpy arrays."""
    return np.max(np.abs((X1.ravel() - X2.ravel())/X1.ravel()))

class TestKalmanTV(unittest.TestCase):
    def setUp(self):
        self.n_meas = np.random.randint(4) + 1
        self.n_state = self.n_meas + np.random.randint(5)
        self.ktv1 = KalmanTV_py(self.n_meas, self.n_state)
    
    def test_predict(self):
        mu_state_past = rand_vec(self.n_state)
        var_state_past = rand_mat(self.n_state)
        mu_state = rand_vec(self.n_state)
        wgt_state = rand_mat(self.n_state, pd=False)
        var_state = rand_mat(self.n_state)
        mu_state_pred1 = np.empty(self.n_state)
        var_state_pred1 = np.empty((self.n_state, self.n_state), order='F')
        self.ktv1.predict(mu_state_pred1, var_state_pred1,
                          mu_state_past, var_state_past,
                          mu_state, wgt_state, var_state)
        mu_state_pred2, var_state_pred2 = \
            predict(mu_state_past, var_state_past,
                    mu_state, wgt_state, var_state)
        self.assertAlmostEqual(rel_err(mu_state_pred1, mu_state_pred2), 0.0, places=7)
        self.assertAlmostEqual(rel_err(var_state_pred1, var_state_pred2), 0.0, places=7)

    def test_update(self):
        mu_state_pred = rand_vec(self.n_state)
        var_state_pred = rand_mat(self.n_state)
        x_meas = rand_vec(self.n_meas)
        mu_meas = rand_vec(self.n_meas)
        wgt_meas = rand_mat(self.n_meas, self.n_state, pd=False)
        var_meas = rand_mat(self.n_meas)
        mu_state_filt1 = np.empty(self.n_state)
        var_state_filt1 = np.empty((self.n_state, self.n_state), order='F')
        self.ktv1.update(mu_state_filt1, var_state_filt1,
                         mu_state_pred, var_state_pred,
                         x_meas, mu_meas, wgt_meas, var_meas)
        mu_state_filt2, var_state_filt2 = \
            update(mu_state_pred, var_state_pred,
                   x_meas, mu_meas, wgt_meas, var_meas)
        self.assertAlmostEqual(rel_err(mu_state_filt1, mu_state_filt2), 0.0, places=7)
        self.assertAlmostEqual(rel_err(var_state_filt1, var_state_filt2), 0.0, places=7)
    
    def test_filter(self):
        mu_state_past = rand_vec(self.n_state)
        var_state_past = rand_mat(self.n_state)
        mu_state = rand_vec(self.n_state)
        wgt_state = rand_mat(self.n_state, pd=False)
        var_state = rand_mat(self.n_state)
        x_meas = rand_vec(self.n_meas)
        mu_meas = rand_vec(self.n_meas)
        wgt_meas = rand_mat(self.n_meas, self.n_state, pd=False)
        var_meas = rand_mat(self.n_meas)
        mu_state_pred1 = np.empty(self.n_state)
        var_state_pred1 = np.empty((self.n_state, self.n_state), order='F')
        mu_state_filt1 = np.empty(self.n_state)
        var_state_filt1 = np.empty((self.n_state, self.n_state), order='F')
        self.ktv1.filter(mu_state_pred1, var_state_pred1,
                         mu_state_filt1, var_state_filt1,
                         mu_state_past, var_state_past,
                         mu_state, wgt_state, var_state,
                         x_meas, mu_meas, wgt_meas, var_meas)
        mu_state_pred2, var_state_pred2, mu_state_filt2, var_state_filt2 = \
            filter(mu_state_past, var_state_past,
                   mu_state, wgt_state, var_state,
                   x_meas, mu_meas, wgt_meas, var_meas)
        self.assertAlmostEqual(rel_err(mu_state_pred1, mu_state_pred2), 0.0, places=7)
        self.assertAlmostEqual(rel_err(var_state_pred1, var_state_pred2), 0.0, places=7)
        self.assertAlmostEqual(rel_err(mu_state_filt1, mu_state_filt2), 0.0, places=7)
        self.assertAlmostEqual(rel_err(var_state_filt1, var_state_filt2), 0.0, places=7)

    def test_smooth_mv(self):
        mu_state_next = rand_vec(self.n_state)
        var_state_next = rand_mat(self.n_state)
        mu_state_filt = rand_vec(self.n_state)
        var_state_filt = rand_mat(self.n_state)
        mu_state_pred = rand_vec(self.n_state)
        var_state_pred = rand_mat(self.n_state)
        wgt_state = rand_mat(self.n_state, pd=False)
        mu_state_smooth1 = np.empty(self.n_state)
        var_state_smooth1 = np.empty((self.n_state, self.n_state), order='F')
        self.ktv1.smooth_mv(mu_state_smooth1, var_state_smooth1,
                            mu_state_next, var_state_next,
                            mu_state_filt, var_state_filt,
                            mu_state_pred, var_state_pred,
                            wgt_state)
        mu_state_smooth2, var_state_smooth2 = \
            smooth_mv(mu_state_next, var_state_next,
                      mu_state_filt, var_state_filt,
                      mu_state_pred, var_state_pred,
                      wgt_state)
        self.assertAlmostEqual(rel_err(mu_state_smooth1, mu_state_smooth2), 0.0, places=7)
        self.assertAlmostEqual(rel_err(var_state_smooth1, var_state_smooth2), 0.0, places=7)

    def test_smooth_sim(self):
        mu_state_past = rand_vec(self.n_state)
        var_state_past = rand_mat(self.n_state)
        x_state_next = rand_vec(self.n_state)
        mu_state = rand_vec(self.n_state)
        wgt_state = 0.01*rand_mat(self.n_state, pd=False)
        var_state = rand_mat(self.n_state)
        z_state = rand_vec(self.n_state)
        x_meas = rand_vec(self.n_meas)
        mu_meas = rand_vec(self.n_meas)
        wgt_meas = rand_mat(self.n_meas, self.n_state, pd=False)
        var_meas = rand_mat(self.n_meas)
        mu_state_pred1 = np.empty(self.n_state)
        var_state_pred1 = np.empty((self.n_state, self.n_state), order='F')
        mu_state_filt1 = np.empty(self.n_state)
        var_state_filt1 = np.empty((self.n_state, self.n_state), order='F')
        x_state_smooth1 = np.empty(self.n_state)
        self.ktv1.filter(mu_state_pred1, var_state_pred1,
                         mu_state_filt1, var_state_filt1,
                         mu_state_past, var_state_past,
                         mu_state, wgt_state, var_state,
                         x_meas, mu_meas, wgt_meas, var_meas)
        self.ktv1.smooth_sim(x_state_smooth1, x_state_next,
                             mu_state_filt1, var_state_filt1,
                             mu_state_pred1, var_state_pred1,
                             wgt_state, z_state)

        mu_state_pred2, var_state_pred2, mu_state_filt2, var_state_filt2 = \
            filter(mu_state_past, var_state_past,
                   mu_state, wgt_state, var_state,
                   x_meas, mu_meas, wgt_meas, var_meas)
        x_state_smooth2 = \
            smooth_sim(x_state_next,
                       mu_state_filt2, var_state_filt2,
                       mu_state_pred2, var_state_pred2,
                       wgt_state, z_state)
        
        self.assertAlmostEqual(rel_err(x_state_smooth1, x_state_smooth2), 0.0, places=7)


    def test_smooth(self):
        mu_state_past = rand_vec(self.n_state)
        var_state_past = rand_mat(self.n_state)
        x_state_next = rand_vec(self.n_state)
        mu_state_next = rand_vec(self.n_state)
        var_state_next = rand_mat(self.n_state)
        mu_state = rand_vec(self.n_state)
        wgt_state = 0.01*rand_mat(self.n_state, pd=False)
        var_state = rand_mat(self.n_state)
        z_state = rand_vec(self.n_state)
        x_meas = rand_vec(self.n_meas)
        mu_meas = rand_vec(self.n_meas)
        wgt_meas = rand_mat(self.n_meas, self.n_state, pd=False)
        var_meas = rand_mat(self.n_meas)
        mu_state_pred1 = np.empty(self.n_state)
        var_state_pred1 = np.empty((self.n_state, self.n_state), order='F')
        mu_state_filt1 = np.empty(self.n_state)
        var_state_filt1 = np.empty((self.n_state, self.n_state), order='F')
        x_state_smooth1 = np.empty(self.n_state)
        mu_state_smooth1 = np.empty(self.n_state)
        var_state_smooth1 = np.empty((self.n_state, self.n_state), order='F')
        self.ktv1.filter(mu_state_pred1, var_state_pred1,
                         mu_state_filt1, var_state_filt1,
                         mu_state_past, var_state_past,
                         mu_state, wgt_state, var_state,
                         x_meas, mu_meas, wgt_meas, var_meas)
        self.ktv1.smooth(x_state_smooth1, mu_state_smooth1,
                         var_state_smooth1, x_state_next,
                         mu_state_next, var_state_next,
                         mu_state_filt1, var_state_filt1,
                         mu_state_pred1, var_state_pred1,
                         wgt_state, z_state)
        mu_state_pred2, var_state_pred2, mu_state_filt2, var_state_filt2 = \
            filter(mu_state_past, var_state_past,
                   mu_state, wgt_state, var_state,
                   x_meas, mu_meas, wgt_meas, var_meas)
        
        x_state_smooth2, mu_state_smooth2, var_state_smooth2 = \
            smooth(x_state_next,
                   mu_state_next, var_state_next,
                   mu_state_filt2, var_state_filt2,
                   mu_state_pred2, var_state_pred2,
                   wgt_state, z_state)
        
        self.assertAlmostEqual(rel_err(mu_state_smooth1, mu_state_smooth2), 0.0, places=7)
        self.assertAlmostEqual(rel_err(var_state_smooth1, var_state_smooth2), 0.0, places=7)
        self.assertAlmostEqual(rel_err(x_state_smooth1, x_state_smooth2), 0.0, places=7)

    def test_forecast(self):
        mu_state_pred = rand_vec(self.n_state)
        var_state_pred = rand_mat(self.n_state)
        mu_meas = rand_vec(self.n_meas)
        wgt_meas = rand_mat(self.n_meas, self.n_state, pd=False)
        var_meas = rand_mat(self.n_meas)
        mu_fore1 = np.empty(self.n_meas)
        var_fore1 = np.empty((self.n_meas, self.n_meas), order='F')
        self.ktv1.forecast(mu_fore1, var_fore1,
                           mu_state_pred, var_state_pred,
                           mu_meas, wgt_meas, var_meas)
        mu_fore2, var_fore2 = \
            forecast(mu_state_pred, var_state_pred,
                     mu_meas, wgt_meas, var_meas)

        self.assertAlmostEqual(rel_err(mu_fore1, mu_fore2), 0.0, places=7)
        self.assertAlmostEqual(rel_err(var_fore1, var_fore2), 0.0, places=7)

if __name__ == '__main__':
    unittest.main()
