cdef extern from "KalmanTV_raw.h" namespace "kalmantv":
    cdef cppclass KalmanTV_raw:
        KalmanTV_raw(int, int) except +
        void predict(double * mu_state_pred,
                     double * var_state_pred,
                     const double * mu_state_past,
                     const double * var_state_past,
                     const double * mu_state,
                     const double * wgt_state,
                     const double * var_state)
        void update(double * mu_state_filt,
                    double * var_state_filt,
                    const double * mu_state_pred,
                    const double * var_state_pred,
                    const double * x_meas,
                    const double * mu_meas,
                    const double * wgt_meas,
                    const double * var_meas)
        void filter(double * mu_state_pred,
                    double * var_state_pred,
                    double * mu_state_filt,
                    double * var_state_filt,
                    const double * mu_state_past,
                    const double * var_state_past,
                    const double * mu_state,
                    const double * wgt_state,
                    const double * var_state,
                    const double * x_meas,
                    const double * mu_meas,
                    const double * wgt_meas,
                    const double * var_meas)
        void smooth_mv(double * mu_state_smooth,
                       double * var_state_smooth,
                       const double * mu_state_next,
                       const double * var_state_next,
                       const double * mu_state_filt,
                       const double * var_state_filt,
                       const double * mu_state_pred,
                       const double * var_state_pred,
                       const double * wgt_state)
        void smooth_sim(double * x_state_smooth,
                        const double * x_state_next,
                        const double * mu_state_filt,
                        const double * var_state_filt,
                        const double * mu_state_pred,
                        const double * var_state_pred,
                        const double * wgt_state,
                        const double * z_state)
        void smooth(double * x_state_smooth,
                    double * mu_state_smooth,
                    double * var_state_smooth,
                    const double * x_state_next,
                    const double * mu_state_next,
                    const double * var_state_next,
                    const double * mu_state_filt,
                    const double * var_state_filt,
                    const double * mu_state_pred,
                    const double * var_state_pred,
                    const double * wgt_state,
                    const double * z_state)
        void state_sim(double * x_state,
                       const double * mu_state,
                       const double * var_state,
                       const double * z_state)
        void forecast(double * mu_fore,
                      double * var_fore,
                      const double * mu_state_pred,
                      const double * var_state_pred,
                      const double * mu_meas,
                      const double * wgt_meas,
                      const double * var_meas)
