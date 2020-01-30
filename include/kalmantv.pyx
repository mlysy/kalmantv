from KalmanTV cimport KalmanTV as CKalmanTV

cdef class KalmanTV:
    cdef CKalmanTV * ktv

    def __cinit__(self, int nMeas, int nState):
        self.ktv = new CKalmanTV(nMeas, nState)

    def __dealloc__(self):
        del self.ktv

    def printX(self):
        self.ktv.printX()

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
    
    def filter(self,
               double[::1] muState_pred,
               double[::1, :] varState_pred,
               double[::1] muState_filt,
               double[::1, :] varState_filt,
               const double[::1] muState_past,
               const double[::1, :] varState_past,
               const double[::1] muState,
               const double[::1, :] wgtState,
               const double[::1, :] varState,
               const double[::1] xMeas,
               const double[::1] muMeas,
               const double[::1, :] wgtMeas,
               const double[::1, :] varMeas):
        self.ktv.filter(& muState_pred[0], & varState_pred[0, 0],
                        & muState_filt[0], & varState_filt[0, 0],
                        & muState_past[0], & varState_past[0, 0],
                        & muState[0], & wgtState[0, 0], & varState[0, 0],
                        & xMeas[0], & muMeas[0],
                        & wgtMeas[0, 0], & varMeas[0, 0])
        return

    def smooth_mv(self,
                  double[::1] muState_smooth,
                  double[::1, :] varState_smooth,
                  const double[::1] muState_next,
                  const double[::1, :] varState_next,
                  const double[::1] muState_filt,
                  const double[::1, :] varState_filt,
                  const double[::1] muState_pred,
                  const double[::1, :] varState_pred,
                  const double[::1, :] wgtState):
        self.ktv.smooth_mv(& muState_smooth[0], & varState_smooth[0, 0],
                           & muState_next[0], & varState_next[0, 0],
                           & muState_filt[0], & varState_filt[0, 0],
                           & muState_pred[0], & varState_pred[0, 0],
                           & wgtState[0, 0])
        return

    def smooth_sim(self,
                   double[::1] xState_smooth,
                   const double[::1] xState_next,
                   const double[::1] muState_filt,
                   const double[::1, :] varState_filt,
                   const double[::1] muState_pred,
                   const double[::1, :] varState_pred,
                   const double[::1, :] wgtState,
                   const double[::1] randState):
        self.ktv.smooth_sim(& xState_smooth[0], & xState_next[0],
                            & muState_filt[0], & varState_filt[0, 0],
                            & muState_pred[0], & varState_pred[0, 0],
                            & wgtState[0, 0], & randState[0])
        return

    def smooth(self,
               double[::1] xState_smooth,
               double[::1] muState_smooth,
               double[::1, :] varState_smooth,
               const double[::1] xState_next,
               const double[::1] muState_next,
               const double[::1, :] varState_next,
               const double[::1] muState_filt,
               const double[::1, :] varState_filt,
               const double[::1] muState_pred,
               const double[::1, :] varState_pred,
               const double[::1, :] wgtState,
               const double[::1] randState):
        self.ktv.smooth(& xState_smooth[0], & muState_smooth[0],
                        & varState_smooth[0, 0], & xState_next[0],
                        & muState_next[0], & varState_next[0, 0],
                        & muState_filt[0], & varState_filt[0, 0],
                        & muState_pred[0], & varState_pred[0, 0],
                        & wgtState[0, 0], & randState[0])
        return
