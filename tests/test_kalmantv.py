import unittest
import numpy as np
import scipy as sp
import scipy.linalg
from kalmantv.cython.kalmantv import PyKalmanTV


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


# pure python implementation (FIXME: use class instead)
def predict(muState_past, varState_past,
            muState, wgtState, varState):
    muState_pred = np.matmul(wgtState, muState_past) + muState
    varState_pred = np.matmul(wgtState, varState_past)
    varState_pred = np.matmul(varState_pred, wgtState.T) + varState
    return muState_pred, varState_pred


def update(muState_pred, varState_pred,
           xMeas, muMeas, wgtMeas, varMeas):
    muMeas_pred = np.matmul(wgtMeas, muState_pred) + muMeas
    varMeasState_pred = np.matmul(wgtMeas, varState_pred)
    varMeasMeas_pred = np.linalg.multi_dot(
        [wgtMeas, varState_pred, wgtMeas.T]) + varMeas
    varStateMeas_pred = np.matmul(varState_pred, wgtMeas.T)
    varState_temp = np.linalg.solve(varMeasMeas_pred, varStateMeas_pred.T).T
    muState_filt = muState_pred + np.matmul(varState_temp, xMeas - muMeas_pred)
    varState_filt = varState_pred - np.matmul(varState_temp, varMeasState_pred)
    return muState_filt, varState_filt

# # check calculation
# nMeas = 2
# nState = 3
# muState_past = rand_vec(nState)
# varState_past = rand_mat(nState)
# muState = rand_vec(nState)
# wgtState = rand_mat(nState, pd=False)
# varState = rand_mat(nState)

# muState_pred, varState_pred = predict(muState_past, varState_past,
#                                       muState, wgtState, varState)

# ktv = PyKalmanTV(nMeas, nState)
# muState_pred2 = np.empty(nState)
# varState_pred2 = np.empty((nState, nState), order='F')
# ktv.predict(muState_pred2, varState_pred2,
#             muState_past, varState_past,
#             muState, wgtState, varState)

# muState_pred - muState_pred2
# varState_pred - varState_pred2

# test suite


class KalmanTVTest(unittest.TestCase):
    def test_predict(self):
        nMeas = np.random.randint(5)
        nState = nMeas + np.random.randint(5)
        muState_past = rand_vec(nState)
        varState_past = rand_mat(nState)
        muState = rand_vec(nState)
        wgtState = rand_mat(nState, pd=False)
        varState = rand_mat(nState)
        # pure python
        muState_pred, varState_pred = predict(muState_past, varState_past,
                                              muState, wgtState, varState)
        # cython
        ktv = PyKalmanTV(nMeas, nState)
        muState_pred2 = np.empty(nState)
        varState_pred2 = np.empty((nState, nState), order='F')
        ktv.predict(muState_pred2, varState_pred2,
                    muState_past, varState_past,
                    muState, wgtState, varState)
        self.assertAlmostEqual(rel_err(muState_pred, muState_pred2), 0.0)
        self.assertAlmostEqual(rel_err(varState_pred, varState_pred2), 0.0)

    def test_update(self):
        nMeas = np.random.randint(5) + 1
        nState = nMeas + np.random.randint(5)
        muState_pred = rand_vec(nState)
        varState_pred = rand_mat(nState)
        xMeas = rand_vec(nMeas)
        muMeas = rand_vec(nMeas)
        wgtMeas = rand_mat(nMeas, nState, pd=False)
        varMeas = rand_mat(nMeas)
        # pure python
        muState_filt, varState_filt = update(muState_pred, varState_pred,
                                             xMeas, muMeas, wgtMeas, varMeas)
        # cython
        ktv = PyKalmanTV(nMeas, nState)
        muState_filt2 = np.empty(nState)
        varState_filt2 = np.empty((nState, nState), order='F')
        ktv.update(muState_filt2, varState_filt2,
                   muState_pred, varState_pred,
                   xMeas, muMeas, wgtMeas, varMeas)
        self.assertAlmostEqual(rel_err(muState_filt, muState_filt2), 0.0)
        self.assertAlmostEqual(rel_err(varState_filt, varState_filt2), 0.0)


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
