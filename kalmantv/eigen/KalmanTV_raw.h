/// @file KalmanTV_raw.h

#ifndef KalmanTV_RAW_h
#define KalmanTV_RAW_h 1

// #undef NDEBUG
// #define EIGEN_RUNTIME_NO_MALLOC
#include <Eigen/Dense>
#include <iostream>
#include "KalmanTV.h"

namespace kalmantv_raw {
  using namespace kalmantv;

  /// Raw buffer for KalmanTV class
  class KalmanTV_raw {
  private:
    int n_meas_; ///< Number of measurement dimensions.
    int n_state_; ///< Number of state dimensions.
    KalmanTV *ktv_; // pointer to internal KalmanTV object with Eigen interface
  public:
    /// Typedefs
    typedef Ref<VectorXd> RefVectorXd;
    typedef const Ref<const VectorXd> cRefVectorXd;
    typedef Ref<MatrixXd> RefMatrixXd;
    typedef const Ref<const MatrixXd> cRefMatrixXd;
    typedef Map<VectorXd> MapVectorXd;
    typedef Map<const VectorXd> cMapVectorXd;
    typedef Map<MatrixXd> MapMatrixXd;
    typedef Map<const MatrixXd> cMapMatrixXd;
    
    KalmanTV_raw() {}
    KalmanTV_raw(int n_meas, int n_state);

    ~KalmanTV_raw();

    /// Raw buffer equivalent.
    void predict(double* mu_state_pred,
                 double* var_state_pred,
                 const double* mu_state_past,
                 const double* var_state_past,
                 const double* mu_state,
                 const double* wgt_state,
                 const double* var_state);
    void update(double* mu_state_filt,
                double* var_state_filt,
                const double* mu_state_pred,
                const double* var_state_pred,
                const double* x_meas,
                const double* mu_meas,
                const double* wgt_meas,
                const double* var_meas);
    void filter(double* mu_state_pred,
                double* var_state_pred,
                double* mu_state_filt,
                double* var_state_filt,
                const double* mu_state_past,
                const double* var_state_past,
                const double* mu_state,
                const double* wgt_state,
                const double* var_state,
                const double* x_meas,
                const double* mu_meas,
                const double* wgt_meas,
                const double* var_meas);
    void smooth_mv(double* mu_state_smooth,
                   double* var_state_smooth,
                   const double* mu_state_next,
                   const double* var_state_next,
                   const double* mu_state_filt,
                   const double* var_state_filt,
                   const double* mu_state_pred,
                   const double* var_state_pred,
                   const double* wgt_state);
    void smooth_sim(double* xState_smooth,
                    const double* xState_next,
                    const double* mu_state_filt,
                    const double* var_state_filt,
                    const double* mu_state_pred,
                    const double* var_state_pred,
                    const double* wgt_state,
                    const double* z_state);    
    void smooth(double* xState_smooth,
                double* mu_state_smooth,
                double* var_state_smooth,
                const double* xState_next,
                const double* mu_state_next,
                const double* var_state_next,
                const double* mu_state_filt,
                const double* var_state_filt,
                const double* mu_state_pred,
                const double* var_state_pred,
                const double* wgt_state,
                const double* z_state);
    void state_sim(double* x_state,
                   const double* mu_state,
                   const double* var_state,
                   const double* z_state);
    void forecast(double* mu_fore,
                  double* var_fore,
                  const double* mu_state_pred,
                  const double* var_state_pred,
                  const double* mu_meas,
                  const double* wgt_meas,
                  const double* var_meas);
  };
  
  inline KalmanTV_raw::KalmanTV_raw(int n_meas, int n_state) {
    n_meas_ = n_meas;
    n_state_ = n_state;
    ktv_ = new KalmanTV(n_meas, n_state);
  }

  inline KalmanTV_raw::~KalmanTV_raw() {
    delete ktv_;
  }
  /// @note Arguments updated to be identical to those with `Eigen` types, so we don't need to re-document.
  inline void KalmanTV_raw::predict(double* mu_state_pred,
                                    double* var_state_pred,
                                    const double* mu_state_past,
                                    const double* var_state_past,
                                    const double* mu_state,
                                    const double* wgt_state,
                                    const double* var_state) {
    MapVectorXd mu_state_pred_(mu_state_pred, n_state_);
    MapMatrixXd var_state_pred_(var_state_pred, n_state_, n_state_);
    cMapVectorXd mu_state_past_(mu_state_past, n_state_);
    cMapMatrixXd var_state_past_(var_state_past, n_state_, n_state_);
    cMapVectorXd mu_state_(mu_state, n_state_);
    cMapMatrixXd wgt_state_(wgt_state, n_state_, n_state_);
    cMapMatrixXd var_state_(var_state, n_state_, n_state_);
    ktv_->predict(mu_state_pred_, var_state_pred_,
                  mu_state_past_, var_state_past_,
                  mu_state_, wgt_state_, var_state_);
    return;
  }
  
  inline void KalmanTV_raw::update(double* mu_state_filt,
                                   double* var_state_filt,
                                   const double* mu_state_pred,
                                   const double* var_state_pred,
                                   const double* x_meas,
                                   const double* mu_meas,
                                   const double* wgt_meas,
                                   const double* var_meas) {
    MapVectorXd mu_state_filt_(mu_state_filt, n_state_);
    MapMatrixXd var_state_filt_(var_state_filt, n_state_, n_state_);
    cMapVectorXd mu_state_pred_(mu_state_pred, n_state_);
    cMapMatrixXd var_state_pred_(var_state_pred, n_state_, n_state_);
    cMapVectorXd x_meas_(x_meas, n_meas_);
    cMapVectorXd mu_meas_(mu_meas, n_meas_);
    cMapMatrixXd wgt_meas_(wgt_meas, n_meas_, n_state_);
    cMapMatrixXd var_meas_(var_meas, n_meas_, n_meas_);
    ktv_->update(mu_state_filt_, var_state_filt_,
                 mu_state_pred_, var_state_pred_,
                 x_meas_, mu_meas_, wgt_meas_, var_meas_);
    return;    
  }

  inline void KalmanTV_raw::filter(double* mu_state_pred,
                                   double* var_state_pred,
                                   double* mu_state_filt,
                                   double* var_state_filt,
                                   const double* mu_state_past,
                                   const double* var_state_past,
                                   const double* mu_state,
                                   const double* wgt_state,
                                   const double* var_state,
                                   const double* x_meas,
                                   const double* mu_meas,
                                   const double* wgt_meas,
                                   const double* var_meas) {
    predict(mu_state_pred, var_state_pred,
            mu_state_past, var_state_past,
            mu_state, wgt_state, var_state);
    update(mu_state_filt, var_state_filt,
           mu_state_pred, var_state_pred,
           x_meas, mu_meas, wgt_meas, var_meas);
    return;    
  }

  inline void KalmanTV_raw::smooth_mv(double* mu_state_smooth,
                                      double* var_state_smooth,
                                      const double* mu_state_next,
                                      const double* var_state_next,
                                      const double* mu_state_filt,
                                      const double* var_state_filt,
                                      const double* mu_state_pred,
                                      const double* var_state_pred,
                                      const double* wgt_state) {
    MapVectorXd mu_state_smooth_(mu_state_smooth, n_state_);
    MapMatrixXd var_state_smooth_(var_state_smooth, n_state_, n_state_);
    cMapVectorXd mu_state_next_(mu_state_next, n_state_);
    cMapMatrixXd var_state_next_(var_state_next, n_state_, n_state_);
    cMapVectorXd mu_state_filt_(mu_state_filt, n_state_);
    cMapMatrixXd var_state_filt_(var_state_filt, n_state_, n_state_);
    cMapVectorXd mu_state_pred_(mu_state_pred, n_state_);
    cMapMatrixXd var_state_pred_(var_state_pred, n_state_, n_state_);
    cMapMatrixXd wgt_state_(wgt_state, n_state_, n_state_);
    ktv_->smooth_mv(mu_state_smooth_, var_state_smooth_,
                    mu_state_next_, var_state_next_,
                    mu_state_filt_, var_state_filt_,
                    mu_state_pred_, var_state_pred_, wgt_state_);
    return;
  }
  
  inline void KalmanTV_raw::smooth_sim(double* xState_smooth,
                                       const double* xState_next,
                                       const double* mu_state_filt,
                                       const double* var_state_filt,
                                       const double* mu_state_pred,
                                   const double* var_state_pred,
                                   const double* wgt_state,
                                   const double* z_state) {
    MapVectorXd xState_smooth_(xState_smooth, n_state_);
    cMapVectorXd xState_next_(xState_next, n_state_);
    cMapVectorXd mu_state_filt_(mu_state_filt, n_state_);
    cMapMatrixXd var_state_filt_(var_state_filt, n_state_, n_state_);
    cMapVectorXd mu_state_pred_(mu_state_pred, n_state_);
    cMapMatrixXd var_state_pred_(var_state_pred, n_state_, n_state_);
    cMapMatrixXd wgt_state_(wgt_state, n_state_, n_state_);
    cMapVectorXd z_state_(z_state, n_state_);
    ktv_->smooth_sim(xState_smooth_, xState_next_, 
                     mu_state_filt_, var_state_filt_,
                     mu_state_pred_, var_state_pred_, 
                     wgt_state_, z_state_);
    return;
  }

  inline void KalmanTV_raw::smooth(double* xState_smooth,
                                   double* mu_state_smooth,
                                   double* var_state_smooth,
                                   const double* xState_next,
                                   const double* mu_state_next,
                                   const double* var_state_next,
                                   const double* mu_state_filt,
                                   const double* var_state_filt,
                                   const double* mu_state_pred,
                                   const double* var_state_pred,
                                   const double* wgt_state,
                                   const double* z_state) {
    smooth_mv(mu_state_smooth, var_state_smooth,
              mu_state_next, var_state_next,
              mu_state_filt, var_state_filt,
              mu_state_pred, var_state_pred, wgt_state);
    smooth_sim(xState_smooth, xState_next, 
               mu_state_filt, var_state_filt,
               mu_state_pred, var_state_pred, 
               wgt_state, z_state);
    return;
  }

  /// Raw buffer equivalent.
  inline void KalmanTV_raw::state_sim(double* x_state,
                                      const double* mu_state,
                                      const double* var_state,
                                      const double* z_state) {
    MapVectorXd x_state_(x_state, n_state_);
    cMapVectorXd mu_state_(mu_state, n_state_);
    cMapMatrixXd var_state_(var_state, n_state_, n_state_);
    cMapVectorXd z_state_(z_state, n_state_);
    ktv_->state_sim(x_state_, mu_state_,
                    var_state_, z_state_);
    return;
  }

  /// @note Arguments updated to be identical to those with `Eigen` types, so we don't need to re-document.
  inline void KalmanTV_raw::forecast(double* mu_fore,
                                 double* var_fore,
                                 const double* mu_state_pred,
                                 const double* var_state_pred,
                                 const double* mu_meas,
                                 const double* wgt_meas,
                                 const double* var_meas) {
    MapVectorXd mu_fore_(mu_fore, n_meas_);
    MapMatrixXd var_fore_(var_fore, n_meas_, n_meas_);
    cMapVectorXd mu_state_pred_(mu_state_pred, n_state_);
    cMapMatrixXd var_state_pred_(var_state_pred, n_state_, n_state_);
    cMapVectorXd mu_meas_(mu_meas, n_meas_);
    cMapMatrixXd wgt_meas_(wgt_meas, n_meas_, n_state_);
    cMapMatrixXd var_meas_(var_meas, n_meas_, n_meas_);
    ktv_->forecast(mu_fore_, var_fore_,
                   mu_state_pred_, var_state_pred_,
                   mu_meas_, wgt_meas_, var_meas_);
    return;
  }
} // end namespace KalmanTV_raw

#endif
