import unittest
import numpy as np
from kalmantv.cython import KalmanTV
from .KalmanTV import KalmanTV as KTV_py


# helper functions

def rel_err(X1, X2):
    """Relative error between two numpy arrays."""
    return np.max(np.abs((X1.ravel() - X2.ravel())/X1.ravel()))


def rand_vec(n):
    """Generate a random vector."""
    return np.random.randn(n)


def rand_mat(n, p=None, pd=True):
    """Generate a random matrix, positive definite if `pd = True`."""
    if p is None:
        p = n
    V = np.zeros((n, p), order='F')
    V[:] = np.random.randn(n, p)
    if (p == n) & pd:
        V[:] = np.matmul(V, V.T)
    return V

# # pure python implementation (FIXME: use class instead)
# def predict(mu_state_past, var_state_past,
#             mu_state, wgt_state, var_state):
#     mu_state_pred = np.matmul(wgt_state, mu_state_past) + mu_state
#     var_state_pred = np.matmul(wgt_state, var_state_past)
#     var_state_pred = np.matmul(var_state_pred, wgt_state.T) + var_state
#     return mu_state_pred, var_state_pred

# def update(mu_state_pred, var_state_pred,
#            x_meas, mu_meas, wgt_meas, var_meas):
#     mu_meas_pred = np.matmul(wgt_meas, mu_state_pred) + mu_meas
#     var_meas_state_pred = np.matmul(wgt_meas, var_state_pred)
#     var_meas_meas_pred = np.linalg.multi_dot(
#         [wgt_meas, var_state_pred, wgt_meas.T]) + var_meas
#     var_state_meas_pred = np.matmul(var_state_pred, wgt_meas.T)
#     var_state_temp = np.linalg.solve(var_meas_meas_pred, var_state_meas_pred.T).T
#     mu_state_filt = mu_state_pred + np.matmul(var_state_temp, x_meas - mu_meas_pred)
#     var_state_filt = var_state_pred - np.matmul(var_state_temp, var_meas_state_pred)
#     return mu_state_filt, var_state_filt

# def filter(mu_state_past, var_state_past,
#            mu_state, wgt_state,
#            var_state, x_meas, mu_meas,
#            wgt_meas, var_meas):
#     mu_state_pred, var_state_pred = predict(mu_state_past=mu_state_past,
#                                           var_state_past=var_state_past,
#                                           mu_state=mu_state,
#                                           wgt_state=wgt_state,
#                                           var_state=var_state)
#     mu_state_filt, var_state_filt = update(mu_state_pred=mu_state_pred,
#                                          var_state_pred=var_state_pred,
#                                          x_meas=x_meas,
#                                          mu_meas=mu_meas,
#                                          wgt_meas=wgt_meas,
#                                          var_meas=var_meas)
#     return mu_state_pred, var_state_pred, mu_state_filt, var_state_filt

# def smooth_mv(mu_state_next, var_state_next,
#               mu_state_filt, var_state_filt,
#               mu_state_pred, var_state_pred,
#               wgt_state):
#     """
#     Perform one step of the Kalman mean/variance smoother.
#     Calculates :math:`\\theta_{n|N}` from :math:`\\theta_{n+1|N}`, :math:`\\theta_{n|n}`, and :math:`\\theta_{n+1|n}`.
#     """
#     var_state_temp = var_state_filt.dot(wgt_state.T)
#     var_state_temp_tilde = np.linalg.solve(var_state_pred, var_state_temp.T).T
#     mu_state_smooth = mu_state_filt + \
#         var_state_temp_tilde.dot(mu_state_next - mu_state_pred)
#     var_state_smooth = var_state_filt + np.linalg.multi_dot(
#         [var_state_temp_tilde, (var_state_next - var_state_pred), var_state_temp_tilde.T])
#     return mu_state_smooth, var_state_smooth

# def smooth_sim(x_state_next, mu_state_filt,
#                var_state_filt, mu_state_pred,
#                var_state_pred, wgt_state, z_state):
#     """
#     Perform one step of the Kalman sampling smoother.
#     Calculates a draw :math:`x_{n|N}` from :math:`x_{n+1|N}`, :math:`\\theta_{n|n}`, and :math:`\\theta_{n+1|n}`.
#     """
#     var_state_temp = var_state_filt.dot(wgt_state.T)
#     var_state_temp_tilde = np.linalg.solve(var_state_pred, var_state_temp.T).T
#     mu_state_sim = mu_state_filt + \
#         var_state_temp_tilde.dot(x_state_next - mu_state_pred)
#     var_state_sim = var_state_filt - var_state_temp_tilde.dot(var_state_temp.T)
#     # Make sure it is semi positive definite for testing
#     var_state_sim2 = var_state_sim.dot(var_state_sim.T)
#     x_state_smooth = np.linalg.cholesky(var_state_sim2).dot(
#         z_state)  # Use var_state_sim instead in the real algorithm
#     x_state_smooth += mu_state_sim
#     return x_state_smooth

# def smooth(x_state_next, mu_state_next,
#            var_state_next, mu_state_filt,
#            var_state_filt, mu_state_pred,
#            var_state_pred, wgt_state, z_state):
#     """
#     Perform one step of both Kalman mean/variance and sampling smoothers.
#     Combines :func:`KalmanTV.smooth_mv` and :func:`KalmanTV.smooth_sim` steps to get :math:`x_{n|N}` and 
#     :math:`\\theta_{n|N}` from :math:`\\theta_{n+1|N}`, :math:`\\theta_{n|n}`, and :math:`\\theta_{n+1|n}`.
#     """
#     mu_state_smooth, var_state_smooth = \
#         smooth_mv(mu_state_next=mu_state_next,
#                   var_state_next=var_state_next,
#                   mu_state_filt=mu_state_filt,
#                   var_state_filt=var_state_filt,
#                   mu_state_pred=mu_state_pred,
#                   var_state_pred=var_state_pred,
#                   wgt_state=wgt_state)
#     x_state_smooth = \
#         smooth_sim(x_state_next=x_state_next,
#                    mu_state_filt=mu_state_filt,
#                    var_state_filt=var_state_filt,
#                    mu_state_pred=mu_state_pred,
#                    var_state_pred=var_state_pred,
#                    wgt_state=wgt_state,
#                    z_state=z_state)
#     return mu_state_smooth, var_state_smooth, x_state_smooth

# # check calculation
# n_meas = 2
# n_state = 3
# mu_state_past = rand_vec(n_state)
# var_state_past = rand_mat(n_state)
# mu_state = rand_vec(n_state)
# wgt_state = rand_mat(n_state, pd=False)
# var_state = rand_mat(n_state)

# mu_state_pred, var_state_pred = predict(mu_state_past, var_state_past,
#                                       mu_state, wgt_state, var_state)

# ktv = KalmanTV(n_meas, n_state)
# mu_state_pred2 = np.empty(n_state)
# var_state_pred2 = np.empty((n_state, n_state), order='F')
# ktv.predict(mu_state_pred2, var_state_pred2,
#             mu_state_past, var_state_past,
#             mu_state, wgt_state, var_state)

# mu_state_pred - mu_state_pred2
# var_state_pred - var_state_pred2

# test suite

class KalmanTVTest(unittest.TestCase):
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
        self.assertAlmostEqual(rel_err(var_state_smooth, var_state_smooth2), 0.0)

    def test_smooth_sim(self):
        n_meas = np.random.randint(5) + 4
        n_state = n_meas + np.random.randint(5)
        x_state_next = rand_vec(n_state)
        mu_state_filt = rand_vec(n_state)
        var_state_filt = rand_mat(n_state)
        mu_state_pred = rand_vec(n_state)
        var_state_pred = rand_mat(n_state)
        wgt_state = rand_mat(n_state, pd=False)
        z_state = rand_vec(n_state)
        # pure python
        KFS = KTV_py(n_meas, n_state)
        x_state_smooth = \
            KFS.smooth_sim(x_state_next, mu_state_filt,
                       var_state_filt, mu_state_pred,
                       var_state_pred, wgt_state, z_state)
        # cython
        ktv = KalmanTV(n_meas, n_state)
        x_state_smooth2 = np.empty(n_state)
        ktv.smooth_sim(x_state_smooth2, x_state_next,
                       mu_state_filt, var_state_filt,
                       mu_state_pred, var_state_pred,
                       wgt_state, z_state)
        self.assertAlmostEqual(rel_err(x_state_smooth, x_state_smooth2), 0.0)

    def test_smooth(self):
        n_meas = np.random.randint(5) + 5
        n_state = n_meas + np.random.randint(5)
        x_state_next = rand_vec(n_state)
        mu_state_next = rand_vec(n_state)
        var_state_next = rand_mat(n_state)
        mu_state_filt = rand_vec(n_state)
        var_state_filt = rand_mat(n_state)
        mu_state_pred = rand_vec(n_state)
        var_state_pred = rand_mat(n_state)
        wgt_state = rand_mat(n_state, pd=False)
        z_state = rand_vec(n_state)
        # pure python
        KFS = KTV_py(n_meas, n_state)
        mu_state_smooth, var_state_smooth, x_state_smooth = \
            KFS.smooth(x_state_next, mu_state_next,
                   var_state_next, mu_state_filt,
                   var_state_filt, mu_state_pred,
                   var_state_pred, wgt_state, z_state)
        # cython
        ktv = KalmanTV(n_meas, n_state)
        x_state_smooth2 = np.empty(n_state)
        mu_state_smooth2 = np.empty(n_state)
        var_state_smooth2 = np.empty((n_state, n_state), order='F')
        ktv.smooth(x_state_smooth2, mu_state_smooth2,
                   var_state_smooth2, x_state_next,
                   mu_state_next, var_state_next,
                   mu_state_filt, var_state_filt,
                   mu_state_pred, var_state_pred,
                   wgt_state, z_state)
        self.assertAlmostEqual(rel_err(mu_state_smooth, mu_state_smooth2), 0.0)
        self.assertAlmostEqual(rel_err(var_state_smooth, var_state_smooth2), 0.0)
        self.assertAlmostEqual(rel_err(x_state_smooth, x_state_smooth2), 0.0)


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
