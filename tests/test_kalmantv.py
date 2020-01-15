import unittest
import numpy as np
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

def filter(muState_past, varState_past,
           muState, wgtState,
           varState, xMeas, muMeas,
           wgtMeas, varMeas):
    muState_pred, varState_pred = predict(muState_past = muState_past, 
                                          varState_past = varState_past,
                                          muState = muState,
                                          wgtState = wgtState,
                                          varState = varState)
    muState_filt, varState_filt = update(muState_pred = muState_pred,
                                         varState_pred = varState_pred,
                                         xMeas = xMeas,
                                         muMeas = muMeas,
                                         wgtMeas = wgtMeas,
                                         varMeas = varMeas)
    return muState_pred, varState_pred, muState_filt, varState_filt

def smooth_mv(muState_next, varState_next,
              muState_filt, varState_filt,
              muState_pred, varState_pred,
              wgtState):
    """
    Perform one step of the Kalman mean/variance smoother.
    Calculates :math:`\\theta_{n|N}` from :math:`\\theta_{n+1|N}`, :math:`\\theta_{n|n}`, and :math:`\\theta_{n+1|n}`.
    """
    varState_temp = varState_filt.dot(wgtState.T)
    varState_temp_tilde = np.linalg.solve(varState_pred, varState_temp.T).T
    muState_smooth = muState_filt + varState_temp_tilde.dot(muState_next - muState_pred)
    varState_smooth = varState_filt + np.linalg.multi_dot([varState_temp_tilde, (varState_next - varState_pred), varState_temp_tilde.T])
    return muState_smooth, varState_smooth

def smooth_sim(xState_next, muState_filt,
               varState_filt, muState_pred,
               varState_pred, wgtState):
    """
    Perform one step of the Kalman sampling smoother.
    Calculates a draw :math:`x_{n|N}` from :math:`x_{n+1|N}`, :math:`\\theta_{n|n}`, and :math:`\\theta_{n+1|n}`.
    """
    varState_temp = varState_filt.dot(wgtState.T)
    varState_temp_tilde = np.linalg.solve(varState_pred, varState_temp.T).T
    muState_sim = muState_filt + varState_temp_tilde.dot(xState_next - muState_pred)
    varState_sim = varState_filt - varState_temp_tilde.dot(varState_temp.T)
    varState_sim2 = varState_sim.dot(varState_sim.T) # Make sure it is semi positive definite
    xState_smooth = np.random.multivariate_normal(muState_sim, varState_sim2)
    return xState_smooth, muState_sim, varState_sim2

def smooth(xState_next, muState_next,
           varState_next, muState_filt,
           varState_filt, muState_pred,
           varState_pred, wgtState):
    """
    Perform one step of both Kalman mean/variance and sampling smoothers.
    Combines :func:`KalmanTV.smooth_mv` and :func:`KalmanTV.smooth_sim` steps to get :math:`x_{n|N}` and 
    :math:`\\theta_{n|N}` from :math:`\\theta_{n+1|N}`, :math:`\\theta_{n|n}`, and :math:`\\theta_{n+1|n}`.
    """
    muState_smooth, varState_smooth = smooth_mv(muState_next = muState_next,
                                                varState_next = varState_next,
                                                muState_filt = muState_filt,
                                                varState_filt = varState_filt,
                                                muState_pred = muState_pred,
                                                varState_pred = varState_pred,
                                                wgtState = wgtState) 
    xState_smooth, muState_sim, varState_sim = smooth_sim(xState_next = xState_next,
                                                          muState_filt = muState_filt,
                                                          varState_filt = varState_filt,
                                                          muState_pred = muState_pred,
                                                          varState_pred = varState_pred,
                                                          wgtState = wgtState)
    return muState_smooth, varState_smooth, xState_smooth, muState_sim, varState_sim
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
    
    def test_filter(self):
        nMeas = np.random.randint(5) + 2
        nState = nMeas + np.random.randint(5)
        muState_past = rand_vec(nState)
        varState_past = rand_mat(nState)
        muState = rand_vec(nState)
        wgtState = rand_mat(nState, pd=False)
        varState = rand_mat(nState)
        xMeas = rand_vec(nMeas)
        muMeas = rand_vec(nMeas)
        wgtMeas = rand_mat(nMeas, nState, pd=False)
        varMeas = rand_mat(nMeas)
        # pure python
        muState_pred, varState_pred, muState_filt, varState_filt = (
            filter(muState_past, varState_past,
                   muState, wgtState,
                   varState, xMeas, muMeas, 
                   wgtMeas, varMeas)
        )
        # cython
        ktv = PyKalmanTV(nMeas, nState)
        muState_pred2 = np.empty(nState)
        varState_pred2 = np.empty((nState, nState), order='F')
        muState_filt2 = np.empty(nState)
        varState_filt2 = np.empty((nState, nState), order='F')
        ktv.filter(muState_pred2, varState_pred2,
                   muState_filt2, varState_filt2,
                   muState_past, varState_past,
                   muState, wgtState, varState, 
                   xMeas, muMeas, wgtMeas, varMeas)
        
        self.assertAlmostEqual(rel_err(muState_pred, muState_pred2), 0.0)
        self.assertAlmostEqual(rel_err(varState_pred, varState_pred2), 0.0)
        self.assertAlmostEqual(rel_err(muState_filt, muState_filt2), 0.0)
        self.assertAlmostEqual(rel_err(varState_filt, varState_filt2), 0.0)
    
    def test_smooth_mv(self):
        nMeas = np.random.randint(5) + 3
        nState = nMeas + np.random.randint(5)
        muState_next = rand_vec(nState)
        varState_next = rand_mat(nState)
        muState_filt = rand_vec(nState)
        varState_filt = rand_mat(nState)
        muState_pred = rand_vec(nState)
        varState_pred = rand_mat(nState)
        wgtState = rand_mat(nState, pd=False)
        # pure python
        muState_smooth, varState_smooth = smooth_mv(muState_next, varState_next,
                                                    muState_filt, varState_filt,
                                                    muState_pred, varState_pred,
                                                    wgtState)
        # cython
        ktv = PyKalmanTV(nMeas, nState)
        muState_smooth2 = np.empty(nState)
        varState_smooth2 = np.empty((nState, nState), order='F')
        ktv.smooth_mv(muState_smooth2, varState_smooth2,
                   muState_next, varState_next,
                   muState_filt, varState_filt,
                   muState_pred, varState_pred,
                   wgtState)
        self.assertAlmostEqual(rel_err(muState_smooth, muState_smooth2), 0.0)
        self.assertAlmostEqual(rel_err(varState_smooth, varState_smooth2), 0.0)
    
    def test_smooth_sim(self):
        nMeas = np.random.randint(5) + 4
        nState = nMeas + np.random.randint(5)
        xState_next = rand_vec(nState)
        muState_filt = rand_vec(nState)
        varState_filt = rand_mat(nState)
        muState_pred = rand_vec(nState)
        varState_pred = rand_mat(nState)
        wgtState = rand_mat(nState, pd=False)
        # pure python
        xState_smooth, muState_sim, varState_sim = smooth_sim(xState_next, muState_filt, 
                                                              varState_filt, muState_pred, 
                                                              varState_pred, wgtState)
        # cython
        ktv = PyKalmanTV(nMeas, nState)
        xState_smooth2 = np.empty(nState)
        muState_sim2 = np.empty(nState)
        varState_sim2 = np.empty((nState, nState), order='F')
        ktv.smooth_sim(xState_smooth2, muState_sim2,
                      varState_sim2, xState_next,
                      muState_filt, varState_filt,
                      muState_pred, varState_pred,
                      wgtState)
        self.assertAlmostEqual(rel_err(muState_sim, muState_sim2), 0.0)
        self.assertAlmostEqual(rel_err(varState_sim, varState_sim2), 0.0)
    
    def test_smooth(self):
        nMeas = np.random.randint(5) + 5
        nState = nMeas + np.random.randint(5)
        xState_next = rand_vec(nState)
        muState_next = rand_vec(nState)
        varState_next = rand_mat(nState)
        muState_filt = rand_vec(nState)
        varState_filt = rand_mat(nState)
        muState_pred = rand_vec(nState)
        varState_pred = rand_mat(nState)
        wgtState = rand_mat(nState, pd=False)
        # pure python
        muState_smooth, varState_smooth, xState_smooth, muState_sim, varState_sim = (
            smooth(xState_next, muState_next,
                   varState_next, muState_filt, 
                   varState_filt, muState_pred, 
                   varState_pred, wgtState)
        )
        # cython
        ktv = PyKalmanTV(nMeas, nState)
        xState_smooth2 = np.empty(nState)
        muState_sim2 = np.empty(nState)
        varState_sim2 = np.empty((nState, nState), order='F')
        muState_smooth2 = np.empty(nState)
        varState_smooth2 = np.empty((nState, nState), order='F')
        ktv.smooth(xState_smooth2, muState_sim2,
                   varState_sim2, muState_smooth2,
                   varState_smooth2, xState_next,
                   muState_next, varState_next,
                   muState_filt, varState_filt,
                   muState_pred, varState_pred,
                   wgtState)
        self.assertAlmostEqual(rel_err(muState_smooth, muState_smooth2), 0.0)
        self.assertAlmostEqual(rel_err(varState_smooth, varState_smooth2), 0.0)
        self.assertAlmostEqual(rel_err(muState_sim, muState_sim2), 0.0)
        self.assertAlmostEqual(rel_err(varState_sim, varState_sim2), 0.0)

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
