from KalmanTV cimport KalmanTV

cdef class PyKalmanTV:
    cdef KalmanTV * ktv

    def __cinit__(self, int nMeas, int nState):
        self.ktv = new KalmanTV(nMeas, nState)

    def __dealloc__(self):
        del self.ktv

    def predict(self,
                double[::1] muState_pred,
                double[::1, :] varState_pred,
                const double[::1] muState_past,
                const double[::1, :] varState_past,
                const double[::1] muState,
                const double[::1, :] wgtState,
                const double[::1, :] varState):
        self.ktv.predict(& muState_pred[0], & varState_pred[0, 0],
                          & muState_past[0], & varState_past[0, 0],
                          & muState[0], & wgtState[0, 0], & varState[0, 0])
        return

    def update(self,
               double[::1] muState_filt,
               double[::1, :] varState_filt,
               const double[::1] muState_pred,
               const double[::1, :] varState_pred,
               const double[::1] xMeas,
               const double[::1] muMeas,
               const double[::1, :] wgtMeas,
               const double[::1, :] varMeas):
        self.ktv.update(& muState_filt[0], & varState_filt[0, 0],
                         & muState_pred[0], & varState_pred[0, 0],
                         & xMeas[0], & muMeas[0],
                         & wgtMeas[0, 0], & varMeas[0, 0])
        return
