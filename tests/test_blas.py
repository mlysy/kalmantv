import unittest
import numpy as np
import warnings
from utils import *
from kalmantv.cython import *

#from kalmantv.cython import KalmanTV
# from kalmantv import *
# from kalmantv_py import KalmanTV as KTV_py  # our own Python implementation
# from pykalman import standard as pks  # pykalman's Python implementation


# test suite

class TestBlas(unittest.TestCase):
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
        solveV(U, X, V, B)
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
