cpdef void state_sim(double[::1] x_state,
                     double[::1, :] llt_state,
                     const double[::1] mu_state,
                     const double[::1, :] var_state,
                     const double[::1] z_state)

cdef class KalmanTV:
    cdef int n_state, n_meas
    cdef double[::1] tmu_state
    cdef double[::1] tmu_state2
    cdef double[::1, :] tvar_state
    cdef double[::1, :] tvar_state2
    cdef double[::1, :] tvar_state3
    cdef double[::1] tmu_meas
    cdef double[::1, :] tvar_meas
    cdef double[::1, :] twgt_meas
    cdef double[::1, :] twgt_meas2
    cdef double[::1, :] llt_meas
    cdef double[::1, :] llt_state

    cpdef void predict(self,
                       double[::1] mu_state_pred,
                       double[::1, :] var_state_pred,
                       const double[::1] mu_state_past,
                       const double[::1, :] var_state_past,
                       const double[::1] mu_state,
                       const double[::1, :] wgt_state,
                       const double[::1, :] var_state)
    cpdef void update(self,
                      double[::1] mu_state_filt,
                      double[::1, :] var_state_filt,
                      const double[::1] mu_state_pred,
                      const double[::1, :] var_state_pred,
                      const double[::1] x_meas,
                      const double[::1] mu_meas,
                      const double[::1, :] wgt_meas,
                      const double[::1, :] var_meas)
    cpdef void filter(self,
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
                      const double[::1, :] var_meas)
    cpdef void smooth_mv(self,
                         double[::1] mu_state_smooth,
                         double[::1, :] var_state_smooth,
                         const double[::1] mu_state_next,
                         const double[::1, :] var_state_next,
                         const double[::1] mu_state_filt,
                         const double[::1, :] var_state_filt,
                         const double[::1] mu_state_pred,
                         const double[::1, :] var_state_pred,
                         const double[::1, :] wgt_state)
    cpdef void smooth_sim(self,
                          double[::1] x_state_smooth,
                          const double[::1] x_state_next,
                          const double[::1] mu_state_filt,
                          const double[::1, :] var_state_filt,
                          const double[::1] mu_state_pred,
                          const double[::1, :] var_state_pred,
                          const double[::1, :] wgt_state,
                          const double[::1] z_state)
    cpdef void smooth(self,
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
                      const double[::1] z_state)
    cpdef void forecast(self,
                        double[::1] mu_fore,
                        double[::1, :] var_fore,
                        const double[::1] mu_state_pred,
                        const double[::1, :] var_state_pred,
                        const double[::1] mu_meas,
                        const double[::1, :] wgt_meas,
                        const double[::1, :] var_meas)
            