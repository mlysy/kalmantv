import unittest
import numpy as np
import warnings

#from kalmantv.cython import KalmanTV
from kalmantv.kalmantv_blas import KalmanTV
from kalmantv.blas_opt import *
from KalmanTV import KalmanTV as KTV_py
from pykalman import standard as pks

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

# test suite

class KalmanTVTest(unittest.TestCase):
    def test_vec_mat(self):
      M = np.random.randint(5) + 1
      K = np.random.randint(5) + 1
      A = rand_mat(M, K)
      alpha = np.random.rand()
      beta = np.random.rand()
      transA = np.random.choice([b'N', b'T'])
      if transA == b'T':
          x = rand_vec(M)
          y = rand_vec(K)
          ans = alpha*A.T.dot(x) + beta*y
      else:
          x = rand_vec(K)
          y = rand_vec(M) 
          ans = alpha*A.dot(x) + beta*y
      mat_vec_mult(y, transA, alpha, beta, A, x)
      self.assertAlmostEqual(rel_err(ans, y), 0.0)
  
    def test_mat(self):
        M = np.random.randint(5) + 1
        K = np.random.randint(5) + 1
        N = np.random.randint(5) + 1
        C = rand_mat(M, N)
        alpha = np.random.rand()
        beta = np.random.rand()
        transA = np.random.choice([b'N', b'T'])
        transB = np.random.choice([b'N', b'T'])
        if transA == b'T' and transB == b'T':
            A = rand_mat(K, M)
            B = rand_mat(N, K)
            ans = alpha*A.T.dot(B.T) + beta*C
        elif transA == b'T':
            A = rand_mat(K, M)
            B = rand_mat(K, N)
            ans = alpha*A.T.dot(B) + beta*C
        elif transB == b'T':
            A = rand_mat(M, K)
            B = rand_mat(N, K)
            ans = alpha*A.dot(B.T) + beta*C
        else:
            A = rand_mat(M, K)
            B = rand_mat(K, N)
            ans = alpha*A.dot(B) + beta*C
        mat_mult(C, transA, transB, alpha, beta, A, B)
        self.assertAlmostEqual(rel_err(ans, C), 0.0)
    
    def test_solveV(self):
        M = np.random.randint(5) + 1
        N = np.random.randint(5) + 1
        V = rand_mat(M)
        B = rand_mat(M, N)
        ans = np.linalg.pinv(V).dot(B)
        U = np.empty((M, M), order='F')
        X = np.empty((M, N), order='F')
        solveV( U, X, V, B)
        self.assertAlmostEqual(rel_err(ans, X), 0.0)
    
    def test_tri_vec(self):
        M = np.random.randint(5) + 1
        A = rand_mat(M)
        uplo = b'L'
        trans = b'N'
        diag = b'N'
        B = np.tril(A)
        z = rand_vec(M)
        ans = B.dot(z)
        tri_vec_mult(z, uplo, trans, diag, A)
        self.assertAlmostEqual(rel_err(ans, z), 0.0)
        
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
        
        #pure python
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
        #cython
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
        self.assertAlmostEqual(rel_err(var_state_smooth, var_state_smooth2), 0.0)
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
        ktv = KalmanTV(n_meas, n_state)
        x_state2 = np.empty(n_state)
        ktv.state_sim(x_state2, mu_state,
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
        self.assertAlmostEqual(rel_err(var_state_smooth, var_state_smooth2), 0.0)

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
