cdef extern from "KalmanTV.h" namespace "KalmanTV":
    cdef cppclass KalmanTV:
        KalmanTV(int, int) except +
        void predict(double * bmuState_pred,
                     double * bvarState_pred,
                     const double * bmuState_past,
                     const double * bvarState_past,
                     const double * bmuState,
                     const double * bwgtState,
                     const double * bvarState)
        void update(double * muState_filt,
                    double * varState_filt,
                    const double * muState_pred,
                    const double * varState_pred,
                    const double * xMeas,
                    const double * muMeas,
                    const double * wgtMeas,
                    const double * varMeas)
