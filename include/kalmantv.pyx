from KalmanTV cimport KalmanTV as CKalmanTV

cdef class KalmanTV:
    cdef CKalmanTV * ktv

    def __cinit__(self, int n_meas, int n_state):
        self.ktv = new CKalmanTV(n_meas, n_state)

    def __dealloc__(self):
        del self.ktv

    def predict(self,
                double[::1] mu_state_pred,
                double[::1, :] var_state_pred,
                const double[::1] mu_state_past,
                const double[::1, :] var_state_past,
                const double[::1] mu_state,
                const double[::1, :] wgt_state,
                const double[::1, :] var_state):
        self.ktv.predict(& mu_state_pred[0], & var_state_pred[0, 0],
                          & mu_state_past[0], & var_state_past[0, 0],
                          & mu_state[0], & wgt_state[0, 0], & var_state[0, 0])
        return

    def update(self,
               double[::1] mu_state_filt,
               double[::1, :] var_state_filt,
               const double[::1] mu_state_pred,
               const double[::1, :] var_state_pred,
               const double[::1] x_meas,
               const double[::1] mu_meas,
               const double[::1, :] wgt_meas,
               const double[::1, :] var_meas):
        self.ktv.update(& mu_state_filt[0], & var_state_filt[0, 0],
                        & mu_state_pred[0], & var_state_pred[0, 0],
                        & x_meas[0], & mu_meas[0],
                        & wgt_meas[0, 0], & var_meas[0, 0])
        return
    
    def filter(self,
               double[::1] mu_state_pred,
               double[::1, :] var_state_pred,
               double[::1] mu_state_filt,
               double[::1, :] var_state_filt,
               const double[::1] mu_state_past,
               const double[::1, :] var_state_past,
               const double[::1] mu_state,
               const double[::1, :] wgt_state,
               const double[::1, :] var_state,
               const double[::1] x_meas,
               const double[::1] mu_meas,
               const double[::1, :] wgt_meas,
               const double[::1, :] var_meas):
        self.ktv.filter(& mu_state_pred[0], & var_state_pred[0, 0],
                        & mu_state_filt[0], & var_state_filt[0, 0],
                        & mu_state_past[0], & var_state_past[0, 0],
                        & mu_state[0], & wgt_state[0, 0], & var_state[0, 0],
                        & x_meas[0], & mu_meas[0],
                        & wgt_meas[0, 0], & var_meas[0, 0])
        return

    def smooth_mv(self,
                  double[::1] mu_state_smooth,
                  double[::1, :] var_state_smooth,
                  const double[::1] mu_state_next,
                  const double[::1, :] var_state_next,
                  const double[::1] mu_state_filt,
                  const double[::1, :] var_state_filt,
                  const double[::1] mu_state_pred,
                  const double[::1, :] var_state_pred,
                  const double[::1, :] wgt_state):
        self.ktv.smooth_mv(& mu_state_smooth[0], & var_state_smooth[0, 0],
                           & mu_state_next[0], & var_state_next[0, 0],
                           & mu_state_filt[0], & var_state_filt[0, 0],
                           & mu_state_pred[0], & var_state_pred[0, 0],
                           & wgt_state[0, 0])
        return

    def smooth_sim(self,
                   double[::1] x_state_smooth,
                   const double[::1] x_state_next,
                   const double[::1] mu_state_filt,
                   const double[::1, :] var_state_filt,
                   const double[::1] mu_state_pred,
                   const double[::1, :] var_state_pred,
                   const double[::1, :] wgt_state,
                   const double[::1] z_state):
        self.ktv.smooth_sim(& x_state_smooth[0], & x_state_next[0],
                            & mu_state_filt[0], & var_state_filt[0, 0],
                            & mu_state_pred[0], & var_state_pred[0, 0],
                            & wgt_state[0, 0], & z_state[0])
        return

    def smooth(self,
               double[::1] x_state_smooth,
               double[::1] mu_state_smooth,
               double[::1, :] var_state_smooth,
               const double[::1] x_state_next,
               const double[::1] mu_state_next,
               const double[::1, :] var_state_next,
               const double[::1] mu_state_filt,
               const double[::1, :] var_state_filt,
               const double[::1] mu_state_pred,
               const double[::1, :] var_state_pred,
               const double[::1, :] wgt_state,
               const double[::1] z_state):
        self.ktv.smooth(& x_state_smooth[0], & mu_state_smooth[0],
                        & var_state_smooth[0, 0], & x_state_next[0],
                        & mu_state_next[0], & var_state_next[0, 0],
                        & mu_state_filt[0], & var_state_filt[0, 0],
                        & mu_state_pred[0], & var_state_pred[0, 0],
                        & wgt_state[0, 0], & z_state[0])
        return

    def state_sim(self,
                  double[::1] x_state,
                  double[::1] mu_state,
                  double[::1, :] var_state,
                  double[::1] z_state):
        self.ktv.state_sim(& x_state[0], & mu_state[0],
                           & var_state[0, 0], & z_state[0])
        return