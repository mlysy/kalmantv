import unittest
import numpy as np
import warnings
from utils import *

#from kalmantv.cython import KalmanTV
from kalmantv import *
# from kalmantv.blas import *
from kalmantv_py import KalmanTV as KTV_py  # our own Python implementation
from pykalman import standard as pks  # pykalman's Python implementation


# test suite

class TestKalmanTV(unittest.TestCase):

    def test_predict(self):
        n_meas = np.random.randint(5)
        n_state = n_meas + np.random.randint(5)
        mu_state_past = rand_vec(n_state)
        var_state_past = rand_mat(n_state)
        mu_state = rand_vec(n_state)
        wgt_state = rand_mat(n_state, pd=False)
        var_state = rand_mat(n_state)
        # pure python
        KFS = KTV_py(n_meas, n_state)
        mu_state_pred, var_state_pred = KFS.predict(mu_state_past, var_state_past,
                                                    mu_state, wgt_state, var_state)
        # cython
        ktv = KalmanTV(n_meas, n_state)
        mu_state_pred2 = np.empty(n_state)
        var_state_pred2 = np.empty((n_state, n_state), order='F')
        ktv.predict(mu_state_pred2, var_state_pred2,
                    mu_state_past, var_state_past,
                    mu_state, wgt_state, var_state)
        self.assertAlmostEqual(rel_err(mu_state_pred, mu_state_pred2), 0.0)
        self.assertAlmostEqual(rel_err(var_state_pred, var_state_pred2), 0.0)

    def test_update(self):
        n_meas = np.random.randint(5) + 1
        n_state = n_meas + np.random.randint(5)
        mu_state_pred = rand_vec(n_state)
        var_state_pred = rand_mat(n_state)
        x_meas = rand_vec(n_meas)
        mu_meas = rand_vec(n_meas)
        wgt_meas = rand_mat(n_meas, n_state, pd=False)
        var_meas = rand_mat(n_meas)
        # pure python
        KFS = KTV_py(n_meas, n_state)
        mu_state_filt, var_state_filt = KFS.update(mu_state_pred, var_state_pred,
                                                   x_meas, mu_meas, wgt_meas, var_meas)
        # cython
        ktv = KalmanTV(n_meas, n_state)
        mu_state_filt2 = np.empty(n_state)
        var_state_filt2 = np.empty((n_state, n_state), order='F')
        ktv.update(mu_state_filt2, var_state_filt2,
                   mu_state_pred, var_state_pred,
                   x_meas, mu_meas, wgt_meas, var_meas)
        self.assertAlmostEqual(rel_err(mu_state_filt, mu_state_filt2), 0.0)
        self.assertAlmostEqual(rel_err(var_state_filt, var_state_filt2), 0.0)

    def test_filter(self):
        n_meas = np.random.randint(5) + 2
        n_state = n_meas + np.random.randint(5)
        mu_state_past = rand_vec(n_state)
        var_state_past = rand_mat(n_state)
        mu_state = rand_vec(n_state)
        wgt_state = rand_mat(n_state, pd=False)
        var_state = rand_mat(n_state)
        x_meas = rand_vec(n_meas)
        mu_meas = rand_vec(n_meas)
        wgt_meas = rand_mat(n_meas, n_state, pd=False)
        var_meas = rand_mat(n_meas)
        # pure python
        KFS = KTV_py(n_meas, n_state)
        mu_state_pred, var_state_pred, mu_state_filt, var_state_filt = (
            KFS.filter(mu_state_past, var_state_past,
                       mu_state, wgt_state,
                       var_state, x_meas, mu_meas,
                       wgt_meas, var_meas)
        )
        # cython
        ktv = KalmanTV(n_meas, n_state)
        mu_state_pred2 = np.empty(n_state)
        var_state_pred2 = np.empty((n_state, n_state), order='F')
        mu_state_filt2 = np.empty(n_state)
        var_state_filt2 = np.empty((n_state, n_state), order='F')
        ktv.filter(mu_state_pred2, var_state_pred2,
                   mu_state_filt2, var_state_filt2,
                   mu_state_past, var_state_past,
                   mu_state, wgt_state, var_state,
                   x_meas, mu_meas, wgt_meas, var_meas)
        self.assertAlmostEqual(rel_err(mu_state_pred, mu_state_pred2), 0.0)
        self.assertAlmostEqual(rel_err(var_state_pred, var_state_pred2), 0.0)
        self.assertAlmostEqual(rel_err(mu_state_filt, mu_state_filt2), 0.0)
        self.assertAlmostEqual(rel_err(var_state_filt, var_state_filt2), 0.0)

    def test_smooth_mv(self):
        n_meas = np.random.randint(5) + 3
        n_state = n_meas + np.random.randint(5)
        mu_state_next = rand_vec(n_state)
        var_state_next = rand_mat(n_state)
        mu_state_filt = rand_vec(n_state)
        var_state_filt = rand_mat(n_state)
        mu_state_pred = rand_vec(n_state)
        var_state_pred = rand_mat(n_state)
        wgt_state = rand_mat(n_state, pd=False)
        # pure python
        KFS = KTV_py(n_meas, n_state)
        mu_state_smooth, var_state_smooth = KFS.smooth_mv(mu_state_next, var_state_next,
                                                          mu_state_filt, var_state_filt,
                                                          mu_state_pred, var_state_pred,
                                                          wgt_state)
        # cython
        ktv = KalmanTV(n_meas, n_state)
        mu_state_smooth2 = np.empty(n_state)
        var_state_smooth2 = np.empty((n_state, n_state), order='F')
        ktv.smooth_mv(mu_state_smooth2, var_state_smooth2,
                      mu_state_next, var_state_next,
                      mu_state_filt, var_state_filt,
                      mu_state_pred, var_state_pred,
                      wgt_state)
        self.assertAlmostEqual(rel_err(mu_state_smooth, mu_state_smooth2), 0.0)
        self.assertAlmostEqual(
            rel_err(var_state_smooth, var_state_smooth2), 0.0)

    def test_smooth_sim(self):
        n_meas = np.random.randint(5) + 4
        n_state = n_meas + np.random.randint(5)
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
        # pure python
        KFS = KTV_py(n_meas, n_state)
        mu_state_pred, var_state_pred, mu_state_filt, var_state_filt = (
            KFS.filter(mu_state_past, var_state_past,
                       mu_state, wgt_state,
                       var_state, x_meas, mu_meas,
                       wgt_meas, var_meas)
        )
        x_state_smooth = \
            KFS.smooth_sim(x_state_next, mu_state_filt,
                           var_state_filt, mu_state_pred,
                           var_state_pred, wgt_state, z_state)

        # cython
        ktv = KalmanTV(n_meas, n_state)
        mu_state_pred2 = np.empty(n_state)
        var_state_pred2 = np.empty((n_state, n_state), order='F')
        mu_state_filt2 = np.empty(n_state)
        var_state_filt2 = np.empty((n_state, n_state), order='F')
        ktv.filter(mu_state_pred2, var_state_pred2,
                   mu_state_filt2, var_state_filt2,
                   mu_state_past, var_state_past,
                   mu_state, wgt_state, var_state,
                   x_meas, mu_meas, wgt_meas, var_meas)
        x_state_smooth2 = np.empty(n_state)
        ktv.smooth_sim(x_state_smooth2, x_state_next,
                       mu_state_filt2, var_state_filt2,
                       mu_state_pred2, var_state_pred2,
                       wgt_state, z_state)
        self.assertAlmostEqual(rel_err(x_state_smooth, x_state_smooth2), 0.0)

    def test_smooth(self):
        n_meas = np.random.randint(5) + 5
        n_state = n_meas + np.random.randint(5)
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

        # pure python
        KFS = KTV_py(n_meas, n_state)
        mu_state_pred, var_state_pred, mu_state_filt, var_state_filt = (
            KFS.filter(mu_state_past, var_state_past,
                       mu_state, wgt_state,
                       var_state, x_meas, mu_meas,
                       wgt_meas, var_meas)
        )
        mu_state_smooth, var_state_smooth, x_state_smooth = \
            KFS.smooth(x_state_next, mu_state_next,
                       var_state_next, mu_state_filt,
                       var_state_filt, mu_state_pred,
                       var_state_pred, wgt_state, z_state)
        # cython
        ktv = KalmanTV(n_meas, n_state)
        mu_state_pred2 = np.empty(n_state)
        var_state_pred2 = np.empty((n_state, n_state), order='F')
        mu_state_filt2 = np.empty(n_state)
        var_state_filt2 = np.empty((n_state, n_state), order='F')
        ktv.filter(mu_state_pred2, var_state_pred2,
                   mu_state_filt2, var_state_filt2,
                   mu_state_past, var_state_past,
                   mu_state, wgt_state, var_state,
                   x_meas, mu_meas, wgt_meas, var_meas)
        x_state_smooth2 = np.empty(n_state)
        mu_state_smooth2 = np.empty(n_state)
        var_state_smooth2 = np.empty((n_state, n_state), order='F')
        ktv.smooth(x_state_smooth2, mu_state_smooth2,
                   var_state_smooth2, x_state_next,
                   mu_state_next, var_state_next,
                   mu_state_filt2, var_state_filt2,
                   mu_state_pred2, var_state_pred2,
                   wgt_state, z_state)
        self.assertAlmostEqual(rel_err(mu_state_smooth, mu_state_smooth2), 0.0)
        self.assertAlmostEqual(
            rel_err(var_state_smooth, var_state_smooth2), 0.0)
        self.assertAlmostEqual(rel_err(x_state_smooth, x_state_smooth2), 0.0)

    def test_state_sim(self):
        n_meas = np.random.randint(5) + 6
        n_state = n_meas + np.random.randint(5)
        mu_state = rand_vec(n_state)
        var_state = rand_mat(n_state)
        z_state = rand_vec(n_state)
        # pure python
        KFS = KTV_py(n_meas, n_state)
        x_state = \
            KFS.state_sim(mu_state, var_state, z_state)
        # cython
        #ktv = KalmanTV(n_meas, n_state)
        x_state2 = np.empty(n_state)
        # ktv.state_sim(x_state2, mu_state,
        #              var_state, z_state)
        llt_state = np.empty((n_state, n_state), order='F')
        state_sim(x_state2, llt_state, mu_state,
                  var_state, z_state)
        self.assertAlmostEqual(rel_err(x_state, x_state2), 0.0)

    def test_pykalman_predict(self):
        n_meas = np.random.randint(5)
        n_state = n_meas + np.random.randint(5)
        mu_state_past = rand_vec(n_state)
        var_state_past = rand_mat(n_state)
        mu_state = rand_vec(n_state)
        wgt_state = rand_mat(n_state, pd=False)
        var_state = rand_mat(n_state)
        mu_state_pred, var_state_pred = (
            pks._filter_predict(
                wgt_state, var_state,
                mu_state, mu_state_past,
                var_state_past
            )
        )
        ktv = KalmanTV(n_meas, n_state)
        mu_state_pred2 = np.empty(n_state)
        var_state_pred2 = np.empty((n_state, n_state), order='F')
        ktv.predict(mu_state_pred2, var_state_pred2,
                    mu_state_past, var_state_past,
                    mu_state, wgt_state, var_state)
        self.assertAlmostEqual(rel_err(mu_state_pred, mu_state_pred2), 0.0)
        self.assertAlmostEqual(rel_err(var_state_pred, var_state_pred2), 0.0)

    def test_pykalman_update(self):
        n_meas = np.random.randint(5) + 1
        n_state = n_meas + np.random.randint(5)
        mu_state_pred = rand_vec(n_state)
        var_state_pred = rand_mat(n_state)
        x_meas = rand_vec(n_meas)
        mu_meas = rand_vec(n_meas)
        wgt_meas = rand_mat(n_meas, n_state, pd=False)
        var_meas = rand_mat(n_meas)
        _, mu_state_filt, var_state_filt = (
            pks._filter_correct(
                wgt_meas, var_meas,
                mu_meas, mu_state_pred,
                var_state_pred, x_meas
            )
        )
        ktv = KalmanTV(n_meas, n_state)
        mu_state_filt2 = np.empty(n_state)
        var_state_filt2 = np.empty((n_state, n_state), order='F')
        ktv.update(mu_state_filt2, var_state_filt2,
                   mu_state_pred, var_state_pred,
                   x_meas, mu_meas, wgt_meas, var_meas)
        self.assertAlmostEqual(rel_err(mu_state_filt, mu_state_filt2), 0.0)
        self.assertAlmostEqual(rel_err(var_state_filt, var_state_filt2), 0.0)

    def test_pykalman_smooth_mv(self):
        # Turn off beign warning from older version of numpy
        warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
        n_meas = np.random.randint(5) + 3
        n_state = n_meas + np.random.randint(5)
        mu_state_next = rand_vec(n_state)
        var_state_next = rand_mat(n_state)
        mu_state_filt = rand_vec(n_state)
        var_state_filt = rand_mat(n_state)
        mu_state_pred = rand_vec(n_state)
        var_state_pred = rand_mat(n_state)
        wgt_state = rand_mat(n_state, pd=False)
        mu_state_smooth, var_state_smooth, _ = (
            pks._smooth_update(
                wgt_state, mu_state_filt,
                var_state_filt, mu_state_pred,
                var_state_pred, mu_state_next,
                var_state_next)
        )
        ktv = KalmanTV(n_meas, n_state)
        mu_state_smooth2 = np.empty(n_state)
        var_state_smooth2 = np.empty((n_state, n_state), order='F')
        ktv.smooth_mv(mu_state_smooth2, var_state_smooth2,
                      mu_state_next, var_state_next,
                      mu_state_filt, var_state_filt,
                      mu_state_pred, var_state_pred,
                      wgt_state)
        self.assertAlmostEqual(rel_err(mu_state_smooth, mu_state_smooth2), 0.0)
        self.assertAlmostEqual(
            rel_err(var_state_smooth, var_state_smooth2), 0.0)

    def test_mvgaussian(self):
        # gss parameters
        # random values for necessary parameters
        N = 3
        n_meas = np.random.randint(5) + 1
        n_state = n_meas + np.random.randint(5)
        n_gss = n_meas + n_state
        mu_state = rand_vec(n_state)
        wgt_state = rand_mat(n_state, pd=False)
        var_state = rand_mat(n_state)
        mu_meas = rand_vec(n_meas)
        wgt_meas = rand_mat(n_meas, n_state, pd=False)
        var_meas = rand_mat(n_meas)

        # get cholesky of variance matrices
        #chol_state = np.linalg.cholesky(var_state)
        #chol_meas = np.linalg.cholesky(var_meas)

        # stack them since they are the same at each time point
        mu_states = np.stack([mu_state]*(N+1), axis=-1)
        wgt_states = np.stack([wgt_state]*(N+1), axis=-1)
        var_states = np.stack([var_state]*(N+1), axis=-1)
        mu_meass = np.stack([mu_meas]*(N+1), axis=-1)
        wgt_meass = np.stack([wgt_meas]*(N+1), axis=-1)
        var_meass = np.stack([var_meas]*(N+1), axis=-1)

        # get gss parameters
        wgt_gsss, mu_gsss, chol_gsss = ss2gss(
            wgt_states, mu_states, var_states, wgt_meass, mu_meass, var_meass)
        gss_mean, gss_var = mv_gaussian(wgt_gsss, mu_gsss, chol_gsss)

        # Kalman parameters
        mu_state_filts = np.zeros((n_state, N+1), order='F')
        var_state_filts = np.zeros((n_state, n_state, N+1), order='F')
        mu_state_preds = np.zeros((n_state, N+1), order='F')
        var_state_preds = np.zeros((n_state, n_state, N+1), order='F')
        mu_state_smooths = np.zeros((n_state, N+1), order='F')
        var_state_smooths = np.zeros((n_state, n_state, N+1), order='F')
        x_state_smooths = np.zeros((n_state, N+1), order='F')
        z_states = rand_mat(n_state, N+1)
        n = 3
        x_meass = rand_mat(n_meas, n)

        ktv = KalmanTV(n_meas, n_state)
        for t in range(n):
            ktv.filter(mu_state_preds[:, t+1],
                       var_state_preds[:, :, t+1],
                       mu_state_filts[:, t+1],
                       var_state_filts[:, :, t+1],
                       mu_state_filts[:, t],
                       var_state_filts[:, :, t],
                       mu_state,
                       wgt_state,
                       var_state,
                       x_meass[:, t],
                       mu_meas,
                       wgt_meas,
                       var_meas)
        mu_state_smooths[:, n] = mu_state_filts[:, n]
        var_state_smooths[:, :, n] = var_state_filts[:, :, n]
        llt_state = np.empty((n_state, n_state), order='F')
        state_sim(x_state_smooths[:, n],
                  llt_state,
                  mu_state_smooths[:, n],
                  var_state_smooths[:, :, n],
                  z_states[:, n])
        for t in reversed(range(1, n)):
            ktv.smooth(x_state_smooths[:, t],
                       mu_state_smooths[:, t],
                       var_state_smooths[:, :, t],
                       x_state_smooths[:, t+1],
                       mu_state_smooths[:, t+1],
                       var_state_smooths[:, :, t+1],
                       mu_state_filts[:, t],
                       var_state_filts[:, :, t],
                       mu_state_preds[:, t+1],
                       var_state_preds[:, :, t+1],
                       wgt_state,
                       z_states[:, t])

        # filter
        ijoint1 = np.array([False]*len(gss_mean))
        for i in range(n):
            ijoint1[n_gss*i+n_state:n_gss*i+n_state+n_meas] = True
        ijoint1[n_gss*(n-1):n_gss*n] = True
        icond1 = np.array([True]*n_meas*(n-1) + [False]
                          * n_state + [True]*n_meas)
        A, b, V = mvncond(gss_mean[ijoint1],
                          gss_var[np.ix_(ijoint1, ijoint1)], icond1)
        mu_tt_filt = A.dot(x_meass.flatten(order='F')) + b
        Sigma_tt_filt = V
        self.assertAlmostEqual(rel_err(mu_tt_filt, mu_state_filts[:, n]), 0.0)
        self.assertAlmostEqual(
            rel_err(Sigma_tt_filt, var_state_filts[:, :, n]), 0.0)

        # smooth_mv
        ijoint2 = np.array([False]*len(gss_mean))
        for i in range(N):
            ijoint2[n_gss*i+n_state:n_gss*i+n_state+n_meas] = True
        ijoint2[n_gss*(n-2):n_gss*(n-1)] = True
        icond2 = np.array([True]*n_meas*(n-2) + [False]
                          * n_state + [True]*n_meas*2)
        A, b, V = mvncond(gss_mean[ijoint2],
                          gss_var[np.ix_(ijoint2, ijoint2)], icond2)
        mu_tt_smooth = A.dot(x_meass.flatten(order='F')) + b
        Sigma_tt_smooth = V
        self.assertAlmostEqual(
            rel_err(mu_tt_smooth, mu_state_smooths[:, n-1]), 0.0)
        self.assertAlmostEqual(
            rel_err(Sigma_tt_smooth, var_state_smooths[:, :, n-1]), 0.0)

        # smooth_sim
        ijoint3 = np.array([False]*len(gss_mean))
        for i in range(N):
            ijoint3[n_gss*i+n_state:n_gss*i+n_state+n_meas] = True
        ijoint3[n_gss*(n-2):n_gss*n] = True
        icond3 = np.array([True]*n_meas*(n-2) + [False] *
                          n_state + [True]*n_meas + [True]*n_gss)
        A, b, V = mvncond(gss_mean[ijoint3],
                          gss_var[np.ix_(ijoint3, ijoint3)], icond3)
        Y_sim = np.concatenate(
            (x_meass[:, 0:n-1].flatten(order='F'), x_state_smooths[:, n], x_meass[:, n-1]))
        mu_tt = A.dot(Y_sim) + b
        Sigma_tt = V
        x_tt_sim = np.linalg.cholesky(Sigma_tt).dot(z_states[:, n-1]) + mu_tt
        self.assertAlmostEqual(rel_err(x_tt_sim, x_state_smooths[:, n-1]), 0.0)


if __name__ == '__main__':
    unittest.main()


# def suite(ntest):
#     suite = unittest.TestSuite()
#     for ii in range(ntest):
#         suite.addTest(KalmanTVTest('test_predict'))
#     return suite


# if __name__ == '__main__':
#     runner = unittest.TextTestRunner()
#     runner.run(suite(ntest=20))
