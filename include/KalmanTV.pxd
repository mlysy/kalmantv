cdef extern from "KalmanTV.h" namespace "KalmanTV":
    cdef cppclass KalmanTV:
        KalmanTV(int, int) except +
        void printX() except +
        void predict(double * muState_pred,
                     double * varState_pred,
                     const double * muState_past,
                     const double * varState_past,
                     const double * muState,
                     const double * wgtState,
                     const double * varState)
        void update(double * muState_filt,
                    double * varState_filt,
                    const double * muState_pred,
                    const double * varState_pred,
                    const double * xMeas,
                    const double * muMeas,
                    const double * wgtMeas,
                    const double * varMeas)
        void filter(double* muState_pred,
                    double* varState_pred,
                    double* muState_filt,
    		    double* varState_filt,
    		    const double* muState_past,
    		    const double* varState_past,
                    const double* muState,
		    const double* wgtState,
		    const double* varState,
    		    const double* xMeas,
                    const double* muMeas,
                    const double* wgtMeas,
                    const double* varMeas)
        void smooth_mv(double* muState_smooth,
                       double* varState_smooth,
                       const double* muState_next,
                       const double* varState_next,
                       const double* muState_filt,
                       const double* varState_filt,
                       const double* muState_pred,
                       const double* varState_pred,
                       const double* wgtState) 
        void smooth_sim(double* xState_smooth,
                        const double* xState_next,
                        const double* muState_filt,
                        const double* varState_filt,
                        const double* muState_pred,
                        const double* varState_pred,
                        const double* wgtState,
                        const double* randState)
        void smooth(double* xState_smooth,
                    double* muState_smooth,
                    double* varState_smooth,
                    const double* xState_next,
                    const double* muState_next,
                    const double* varState_next,
                    const double* muState_filt,
                    const double* varState_filt,
                    const double* muState_pred,
                    const double* varState_pred,
                    const double* wgtState,
                    const double* randState)
            
