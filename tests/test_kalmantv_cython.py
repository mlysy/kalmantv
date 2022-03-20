import unittest
import numpy as np
import utils
from utils import *
from kalmantv_py import KalmanTV as KalmanTV_py
from kalmantv.cython import KalmanTV as KalmanTV_cy


class TestKalmanTV(unittest.TestCase):
    def setUp(self):
        self.n_meas = np.random.randint(5) + 1
        self.n_state = self.n_meas + np.random.randint(5)
        self.ktv1 = KalmanTV_py(self.n_meas, self.n_state)
        self.ktv2 = KalmanTV_cy(self.n_meas, self.n_state)

    def test_predict(self):
        mu_state_pred1, var_state_pred1, mu_state_pred2, var_state_pred2 = \
            utils._test_predict(self.n_state, self.n_meas,
                                self.ktv1, self.ktv2)
        self.assertAlmostEqual(rel_err(mu_state_pred1, mu_state_pred2), 0.0)
        self.assertAlmostEqual(rel_err(var_state_pred1, var_state_pred2), 0.0)

    def test_update(self):
        mu_state_filt1, var_state_filt1, mu_state_filt2, var_state_filt2 = \
            utils._test_update(self.n_state, self.n_meas,
                               self.ktv1, self.ktv2)
        self.assertAlmostEqual(rel_err(mu_state_filt1, mu_state_filt2), 0.0)
        self.assertAlmostEqual(rel_err(var_state_filt1, var_state_filt2), 0.0)

    def test_filter(self):
        mu_state_pred1, var_state_pred1, mu_state_filt1, var_state_filt1, \
            mu_state_pred2, var_state_pred2, mu_state_filt2, var_state_filt2 = \
            utils._test_filter(self.n_state, self.n_meas,
                               self.ktv1, self.ktv2)
        self.assertAlmostEqual(rel_err(mu_state_pred1, mu_state_pred2), 0.0)
        self.assertAlmostEqual(rel_err(var_state_pred1, var_state_pred2), 0.0)
        self.assertAlmostEqual(rel_err(mu_state_filt1, mu_state_filt2), 0.0)
        self.assertAlmostEqual(rel_err(var_state_filt1, var_state_filt2), 0.0)

    def test_smooth_mv(self):
        mu_state_smooth1, var_state_smooth1, \
            mu_state_smooth2, var_state_smooth2 = \
            utils._test_smooth_mv(self.n_state, self.n_meas,
                                  self.ktv1, self.ktv2)
        self.assertAlmostEqual(
            rel_err(mu_state_smooth1, mu_state_smooth2), 0.0)
        self.assertAlmostEqual(
            rel_err(var_state_smooth1, var_state_smooth2), 0.0)

    def test_smooth_sim(self):
        x_state_smooth1, x_state_smooth2 = \
            utils._test_smooth_sim(self.n_state, self.n_meas,
                                   self.ktv1, self.ktv2)
        self.assertAlmostEqual(rel_err(x_state_smooth1, x_state_smooth2), 0.0)

    def test_smooth(self):
        mu_state_smooth1, var_state_smooth1, x_state_smooth1, \
            mu_state_smooth2, var_state_smooth2, x_state_smooth2, = \
            utils._test_smooth(self.n_state, self.n_meas,
                               self.ktv1, self.ktv2)
        self.assertAlmostEqual(
            rel_err(mu_state_smooth1, mu_state_smooth2), 0.0)
        self.assertAlmostEqual(
            rel_err(var_state_smooth1, var_state_smooth2), 0.0)
        self.assertAlmostEqual(rel_err(x_state_smooth1, x_state_smooth2), 0.0)

    def test_forecast(self):
        mu_fore1, var_fore1, mu_fore2, var_fore2 = \
            utils._test_forecast(self.n_state, self.n_meas,
                                self.ktv1, self.ktv2)
        self.assertAlmostEqual(rel_err(mu_fore1, mu_fore2), 0.0)
        self.assertAlmostEqual(rel_err(var_fore1, var_fore2), 0.0)

if __name__ == '__main__':
    unittest.main()
