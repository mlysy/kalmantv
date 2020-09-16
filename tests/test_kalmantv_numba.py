import unittest
import numpy as np
from utils import *
from kalmantv_py import KalmanTV as KalmanTV_py
from kalmantv_numba import KalmanTV as KalmanTV_nb


class TestKalmanTV(unittest.TestCase):
    def setUp(self):
        self.n_meas = np.random.randint(5) + 1
        self.n_state = self.n_meas + np.random.randint(5)
        self.ktv1 = KalmanTV_py(self.n_meas, self.n_state)
        self.ktv2 = KalmanTV_nb(self.n_meas, self.n_state)

    def test_predict(self):
        mu_state_past = rand_vec(self.n_state)
        var_state_past = rand_mat(self.n_state)
        mu_state = rand_vec(self.n_state)
        wgt_state = rand_mat(self.n_state, pd=False)
        var_state = rand_mat(self.n_state)
        # python
        mu_state_pred1, var_state_pred1 = \
            self.ktv1.predict(mu_state_past, var_state_past,
                              mu_state, wgt_state, var_state)
        # numba
        mu_state_pred2 = np.empty(self.n_state)
        var_state_pred2 = np.empty((self.n_state, self.n_state), order='F')
        self.ktv2.predict(mu_state_pred2, var_state_pred2,
                          mu_state_past, var_state_past,
                          mu_state, wgt_state, var_state)
        self.assertAlmostEqual(rel_err(mu_state_pred1, mu_state_pred2), 0.0)
        self.assertAlmostEqual(rel_err(var_state_pred1, var_state_pred2), 0.0)

    def test_update(self):
        mu_state_pred = rand_vec(self.n_state)
        var_state_pred = rand_mat(self.n_state)
        x_meas = rand_vec(self.n_meas)
        mu_meas = rand_vec(self.n_meas)
        wgt_meas = rand_mat(self.n_meas, self.n_state, pd=False)
        var_meas = rand_mat(self.n_meas)
        # pure python
        mu_state_filt1, var_state_filt1 = \
            self.ktv1.update(mu_state_pred, var_state_pred,
                             x_meas, mu_meas, wgt_meas, var_meas)
        # numba
        mu_state_filt2 = np.empty(self.n_state)
        var_state_filt2 = np.empty((self.n_state, self.n_state), order='F')
        self.ktv2.update(mu_state_filt2, var_state_filt2,
                         mu_state_pred, var_state_pred,
                         x_meas, mu_meas, wgt_meas, var_meas)
        self.assertAlmostEqual(rel_err(mu_state_filt1, mu_state_filt2), 0.0)
        self.assertAlmostEqual(rel_err(var_state_filt1, var_state_filt2), 0.0)

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
        # pure python
        mu_state_pred1, var_state_pred1, mu_state_filt1, var_state_filt1 = \
            self.ktv1.filter(mu_state_past, var_state_past,
                             mu_state, wgt_state,
                             var_state, x_meas, mu_meas,
                             wgt_meas, var_meas)
        # numba
        mu_state_pred2 = np.empty(self.n_state)
        var_state_pred2 = np.empty((self.n_state, self.n_state), order='F')
        mu_state_filt2 = np.empty(self.n_state)
        var_state_filt2 = np.empty((self.n_state, self.n_state), order='F')
        self.ktv2.filter(mu_state_pred2, var_state_pred2,
                         mu_state_filt2, var_state_filt2,
                         mu_state_past, var_state_past,
                         mu_state, wgt_state, var_state,
                         x_meas, mu_meas, wgt_meas, var_meas)
        self.assertAlmostEqual(rel_err(mu_state_pred1, mu_state_pred2), 0.0)
        self.assertAlmostEqual(rel_err(var_state_pred1, var_state_pred2), 0.0)
        self.assertAlmostEqual(rel_err(mu_state_filt1, mu_state_filt2), 0.0)
        self.assertAlmostEqual(rel_err(var_state_filt1, var_state_filt2), 0.0)

    def test_smooth_mv(self):
        mu_state_next = rand_vec(self.n_state)
        var_state_next = rand_mat(self.n_state)
        mu_state_filt = rand_vec(self.n_state)
        var_state_filt = rand_mat(self.n_state)
        mu_state_pred = rand_vec(self.n_state)
        var_state_pred = rand_mat(self.n_state)
        wgt_state = rand_mat(self.n_state, pd=False)
        # pure python
        mu_state_smooth1, var_state_smooth1 = \
            self.ktv1.smooth_mv(mu_state_next, var_state_next,
                                mu_state_filt, var_state_filt,
                                mu_state_pred, var_state_pred,
                                wgt_state)
        # numba
        mu_state_smooth2 = np.empty(self.n_state)
        var_state_smooth2 = np.empty((self.n_state, self.n_state), order='F')
        self.ktv2.smooth_mv(mu_state_smooth2, var_state_smooth2,
                            mu_state_next, var_state_next,
                            mu_state_filt, var_state_filt,
                            mu_state_pred, var_state_pred,
                            wgt_state)
        self.assertAlmostEqual(
            rel_err(mu_state_smooth1, mu_state_smooth2), 0.0)
        self.assertAlmostEqual(
            rel_err(var_state_smooth1, var_state_smooth2), 0.0)

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
        # pure python
        mu_state_pred1, var_state_pred1, mu_state_filt1, var_state_filt1 = \
            self.ktv1.filter(mu_state_past, var_state_past,
                             mu_state, wgt_state,
                             var_state, x_meas, mu_meas,
                             wgt_meas, var_meas)
        x_state_smooth1 = \
            self.ktv1.smooth_sim(x_state_next, mu_state_filt1,
                                 var_state_filt1, mu_state_pred1,
                                 var_state_pred1, wgt_state, z_state)

        # numba
        mu_state_pred2 = np.empty(self.n_state)
        var_state_pred2 = np.empty((self.n_state, self.n_state), order='F')
        mu_state_filt2 = np.empty(self.n_state)
        var_state_filt2 = np.empty((self.n_state, self.n_state), order='F')
        x_state_smooth2 = np.empty(self.n_state)
        self.ktv2.filter(mu_state_pred2, var_state_pred2,
                         mu_state_filt2, var_state_filt2,
                         mu_state_past, var_state_past,
                         mu_state, wgt_state, var_state,
                         x_meas, mu_meas, wgt_meas, var_meas)
        self.ktv2.smooth_sim(x_state_smooth2, x_state_next,
                             mu_state_filt2, var_state_filt2,
                             mu_state_pred2, var_state_pred2,
                             wgt_state, z_state)
        self.assertAlmostEqual(rel_err(x_state_smooth1, x_state_smooth2), 0.0)

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

        # pure python
        mu_state_pred1, var_state_pred1, mu_state_filt1, var_state_filt1 = \
            self.ktv1.filter(mu_state_past, var_state_past,
                             mu_state, wgt_state,
                             var_state, x_meas, mu_meas,
                             wgt_meas, var_meas)
        mu_state_smooth1, var_state_smooth1, x_state_smooth1 = \
            self.ktv1.smooth(x_state_next, mu_state_next,
                             var_state_next, mu_state_filt1,
                             var_state_filt1, mu_state_pred1,
                             var_state_pred1, wgt_state, z_state)
        # numba
        mu_state_pred2 = np.empty(self.n_state)
        var_state_pred2 = np.empty((self.n_state, self.n_state), order='F')
        mu_state_filt2 = np.empty(self.n_state)
        var_state_filt2 = np.empty((self.n_state, self.n_state), order='F')
        x_state_smooth2 = np.empty(self.n_state)
        mu_state_smooth2 = np.empty(self.n_state)
        var_state_smooth2 = np.empty((self.n_state, self.n_state), order='F')
        self.ktv2.filter(mu_state_pred2, var_state_pred2,
                         mu_state_filt2, var_state_filt2,
                         mu_state_past, var_state_past,
                         mu_state, wgt_state, var_state,
                         x_meas, mu_meas, wgt_meas, var_meas)
        self.ktv2.smooth(x_state_smooth2, mu_state_smooth2,
                         var_state_smooth2, x_state_next,
                         mu_state_next, var_state_next,
                         mu_state_filt2, var_state_filt2,
                         mu_state_pred2, var_state_pred2,
                         wgt_state, z_state)
        self.assertAlmostEqual(
            rel_err(mu_state_smooth1, mu_state_smooth2), 0.0)
        self.assertAlmostEqual(
            rel_err(var_state_smooth1, var_state_smooth2), 0.0)
        self.assertAlmostEqual(rel_err(x_state_smooth1, x_state_smooth2), 0.0)


if __name__ == '__main__':
    unittest.main()

# --- scratch ------------------------------------------------------------------

# from numba import njit
# from utils import *

# from kalmantv_numba import KalmanTV
# from kalmantv_py import KalmanTV as KTV_py  # our own Python implementation

# # test predict
# n_meas = np.random.randint(5) + 1
# n_state = n_meas + np.random.randint(5)
# mu_state_past = rand_vec(n_state)
# var_state_past = rand_mat(n_state)
# mu_state = rand_vec(n_state)
# wgt_state = rand_mat(n_state, pd=False)
# var_state = rand_mat(n_state)

# # pure python
# KFS = KTV_py(n_meas, n_state)
# mu_state_pred, var_state_pred = KFS.predict(mu_state_past, var_state_past,
#                                             mu_state, wgt_state, var_state)
# # numba
# ktv = KalmanTV(n_meas, n_state)
# mu_state_pred2 = np.empty(n_state)
# var_state_pred2 = np.empty((n_state, n_state), order='F')
# ktv.predict(mu_state_pred2, var_state_pred2,
#             mu_state_past, var_state_past,
#             mu_state, wgt_state, var_state)

# # test smooth
# n_meas = np.random.randint(5) + 1
# n_state = n_meas + np.random.randint(5)
# mu_state_next = rand_vec(n_state)
# var_state_next = rand_mat(n_state)
# mu_state_filt = rand_vec(n_state)
# var_state_filt = rand_mat(n_state)
# mu_state_pred = rand_vec(n_state)
# var_state_pred = rand_mat(n_state)
# wgt_state = rand_mat(n_state, pd=False)
# # pure python
# ktv1 = KalmanTV_py(n_meas, n_state)
# mu_state_smooth1, var_state_smooth1, var_state_temp_tilde = \
#     ktv1.smooth_mv(mu_state_next, var_state_next,
#                    mu_state_filt, var_state_filt,
#                    mu_state_pred, var_state_pred,
#                    wgt_state)
# # numba
# ktv2 = KalmanTV_nb(n_meas, n_state)
# mu_state_smooth2 = np.empty(n_state)
# var_state_smooth2 = np.empty((n_state, n_state), order='F')
# tvar_state, tvar_state3 = ktv2.smooth_mv(mu_state_smooth2, var_state_smooth2,
#                                          mu_state_next, var_state_next,
#                                          mu_state_filt, var_state_filt,
#                                          mu_state_pred, var_state_pred,
#                                          wgt_state)

# breakpoint()

# print("mu_state_smooth1 = \n", mu_state_smooth1)
# print("mu_state_smooth2 = \n", mu_state_smooth2)
# print("var_state_smooth1 = \n", var_state_smooth1)
# print("var_state_smooth2 = \n", var_state_smooth2)
# print("var_state_temp_tilde = \n", var_state_temp_tilde)
# print("tvar_state = \n", tvar_state)
# print("tvar_state3 = \n", tvar_state3)

# # --- other tests --------------------------------------------------------------

# copy_to_fortran_order = njit(scipy_linalg.ol_copy_to_fortran_order)
