/// @file KalmanTV.h

#ifndef KalmanTV_h
#define KalmanTV_h 1

#include <Eigen/Dense>
#include <random>
// #include <iostream>

namespace KalmanTV {
  using namespace Eigen;

  /// Time-Varying Kalman Filter and Smoother.
  ///
  /// Model is
  ///
  /// `x_n = c_n + T_n x_n-1 + R_n^{1/2} eps_n`
  /// `y_n = d_n + W_n x_n + H_n^{1/2} eta_n`
  ///
  /// Naming conventions:
  ///
  /// - `meas` and `state`.
  /// - `x`, `mu`, `var`, `wgt`.
  /// - `_past`: `n-1|n-1` (filter)
  /// - `_pred`: `n|n-1`
  /// - `_filt`: `n|n`
  /// - `_next`: `n+1|N` (smoother)
  /// - `_smooth`: `n|N`
  /// - `mu_n|m = E[x_n | y_0:m]`
  /// - similarly for `Sigma_n|m` and `theta_n|m = (mu_n|m, Sigma_n|m)`.
  /// - `x_n|m` is a draw from `p(x_n | x_n+1, y_0:m)`.
  ///
  /// So for example we have:
  /// - `x_n = xState[n]`
  /// - `W_n = wgtMeas[n]`
  /// - `E[x_n | y_0:n] = muState_curr`
  /// - `var(x_n | y_0:N) = varState_smooth`
  ///
  class KalmanTV {
  private:
    int nMeas_; ///< Number of measurement dimensions.
    int nState_; ///< Number of state dimensions.
    VectorXd tmuState_; ///< Temporary storage for mean vector.
    VectorXd tmuState2_;
    MatrixXd tvarState_; ///< Temporary storage for variance matrix.
    MatrixXd tvarState2_;
    MatrixXd tvarState3_;
    VectorXd tmuMeas_;
    MatrixXd twgtMeas_;
    MatrixXd twgtMeas2_;
    MatrixXd tvarMeas_;
    LLT<MatrixXd> lltMeas_;
    LLT<MatrixXd> lltState_;
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
    /// Constructor
    KalmanTV(int nMeas, int nState);
    /// Perform one prediction step of the Kalman filter.
    ///
    /// Calculates `theta_n|n-1`from `theta_n-1|n-1`.
    void predict(RefVectorXd muState_pred,
                 RefMatrixXd varState_pred,
                 cRefVectorXd& muState_past,
                 cRefMatrixXd& varState_past,
                 cRefVectorXd& muState,
                 cRefMatrixXd& wgtState,
                 cRefMatrixXd& varState);
    /// Raw buffer equivalent.
    void predict(double* bmuState_pred,
                 double* bvarState_pred,
                 const double* bmuState_past,
                 const double* bvarState_past,
                 const double* bmuState,
                 const double* bwgtState,
                 const double* bvarState);
    /// Perform one update step of the Kalman filter.
    ///
    /// Calculates `theta_n|n` from `theta_n|n-1`.
    void update(RefVectorXd muState_filt,
                RefMatrixXd varState_filt,
                cRefVectorXd& muState_pred,
                cRefMatrixXd& varState_pred,
                cRefVectorXd& xMeas,
                cRefVectorXd& muMeas,
                cRefMatrixXd& wgtMeas,
                cRefMatrixXd& varMeas);
    /// Raw buffer equivalent.
    void update(double* bmuState_filt,
                double* bvarState_filt,
                const double* bmuState_pred,
                const double* bvarState_pred,
                const double* bxMeas,
                const double* bmuMeas,
                const double* bwgtMeas,
                const double* bvarMeas);
    /// Perform one step of the Kalman filter.
    /// Combines `predict` and `update` steps to get `theta_n|n` from `theta_n-1|n-1`.
    // void filter(RefVectorXd muState_pred,
    // 		RefMatrixXd varState_pred,
    //     RefVectorXd muState_filt,
    // 		RefMatrixXd varState_filt,
    // 		cRefVectorXd& muState_past,
    // 		cRefMatrixXd& varState_past,
    // 		cRefVectorXd& muState,
    // 		cRefMatrixXd& wgtState,
    // 		cRefMatrixXd& varState,
    // 		cRefVectorXd& xMeas,
    // 		cRefVectorXd& muMeas,
    // 		cRefMatrixXd& wgtMeas,
    // 		cRefMatrixXd& varMeas);
    /// Raw buffer equivalent.
    void filter(double* bmuState_pred,
                double* bvarState_pred,
                double* bmuState_filt,
                double* bvarState_filt,
                const double* bmuState_past,
                const double* bvarState_past,
                const double* bmuState,
                const double* bwgtState,
                const double* bvarState,
                const double* bxMeas,
                const double* bmuMeas,
                const double* bwgtMeas,
                const double* bvarMeas);
    /// Perform one step of the Kalman mean/variance smoother.
    ///
    /// Calculates `theta_n|N` from `theta_n+1|N`, `theta_n+1|n+1`, and `theta_n+1|n`.  **Is the indexing correct?**
    void smooth_mv(RefVectorXd muState_smooth,
                   RefMatrixXd varState_smooth,
                   cRefVectorXd& muState_next,
                   cRefMatrixXd& varState_next,
                   cRefVectorXd& muState_filt,
                   cRefMatrixXd& varState_filt,
                   cRefVectorXd& muState_pred,
                   cRefMatrixXd& varState_pred,
                   cRefMatrixXd& wgtState);
    /// Raw buffer equivalent.
    void smooth_mv(double* bmuState_smooth,
                   double* bvarState_smooth,
                   const double* bmuState_next,
                   const double* bvarState_next,
                   const double* bmuState_filt,
                   const double* bvarState_filt,
                   const double* bmuState_pred,
                   const double* bvarState_pred,
                   const double* bwgtState);
    /// Perform one step of the Kalman sampling smoother.
    ///
    /// Calculates a draw `x_n|N` from `x_n+1|N`, `theta_n+1|n+1`, and `theta_n+1|n`.  **Is the indexing correct?**
    void smooth_sim(RefVectorXd xState_smooth,
                    RefVectorXd muState_sim,
                    RefMatrixXd varState_sim,
                    cRefVectorXd& xState_next,
                    cRefVectorXd& muState_filt,
                    cRefMatrixXd& varState_filt,
                    cRefVectorXd& muState_pred,
                    cRefMatrixXd& varState_pred,
                    cRefMatrixXd& wgtState);
    /// Raw buffer equivalent.                    
    void smooth_sim(double* bxState_smooth,
                    double* bmuState_sim,
                    double* bvarState_sim,
                    const double* bxState_next,
                    const double* bmuState_filt,
                    const double* bvarState_filt,
                    const double* bmuState_pred,
                    const double* bvarState_pred,
                    const double* bwgtState);    
    // /// Perfrom one step of both Kalman mean/variance and sampling smoothers.
    // void smooth(RefVectorXd xState_smooth,
    // 		RefVectorXd muState_smooth,
    // 		RefMatrixXd varState_smooth,
    // 		cRefVectorXd& muState_next,
    // 		cRefMatrixXd& varState_next,
    // 		cRefVectorXd& muState_filt,
    // 		cRefMatrixXd& varState_filt,
    // 		cRefVectorXd& muState_pred,
    // 		cRefMatrixXd& varState_pred,
    // 		cRefMatrixXd& wgtState);
    /// Raw buffer equivalent.                    
    void smooth(double* bxState_smooth,
                double* bmuState_sim,
                double* bvarState_sim,
                double* bmuState_smooth,
                double* bvarState_smooth,
                const double* bxState_next,
                const double* bmuState_next,
                const double* bvarState_next,
                const double* bmuState_filt,
                const double* bvarState_filt,
                const double* bmuState_pred,
                const double* bvarState_pred,
                const double* bwgtState); 
  };

  /// @param[in] nMeas Number of measurement variables.
  /// @param[in] nState Number of state variables.
  inline KalmanTV::KalmanTV(int nMeas, int nState) {
    // problem dimensions
    nMeas_ = nMeas;
    nState_ = nState;
    // temporary storage
    tmuState_ = VectorXd::Zero(nState_);
    tmuState2_ = VectorXd::Zero(nState_);
    tvarState_ = MatrixXd::Identity(nState_, nState_);
    tvarState2_ = MatrixXd::Identity(nState_, nState_);
    tvarState3_ = MatrixXd::Identity(nState_, nState_);
    tmuMeas_ = VectorXd::Zero(nMeas_);
    tvarMeas_ = MatrixXd::Identity(nMeas_, nMeas_);
    twgtMeas_ = MatrixXd::Zero(nMeas_, nState_);
    twgtMeas2_ = MatrixXd::Zero(nMeas_, nState_);
    // cholesky solvers
    lltMeas_.compute(MatrixXd::Identity(nMeas_, nMeas_));
    lltState_.compute(MatrixXd::Identity(nState_, nState_));
  }

  inline void KalmanTV::predict(RefVectorXd muState_pred,
                                RefMatrixXd varState_pred,
                                cRefVectorXd& muState_past,
                                cRefMatrixXd& varState_past,
                                cRefVectorXd& muState,
                                cRefMatrixXd& wgtState,
                                cRefMatrixXd& varState) {
    // check if this can be done without temporary allocation
    muState_pred.noalias() = wgtState * muState_past + muState;
    // need to assign to temporary for matrix triple product
    tvarState_.noalias() = wgtState * varState_past;
    // temporary allocation?
    varState_pred.noalias() = tvarState_ * wgtState.adjoint() + varState;
    return;
  }
  inline void KalmanTV::predict(double* bmuState_pred,
                                double* bvarState_pred,
                                const double* bmuState_past,
                                const double* bvarState_past,
                                const double* bmuState,
                                const double* bwgtState,
                                const double* bvarState) {
    MapVectorXd muState_pred(bmuState_pred, nState_);
    MapMatrixXd varState_pred(bvarState_pred, nState_, nState_);
    cMapVectorXd muState_past(bmuState_past, nState_);
    cMapMatrixXd varState_past(bvarState_past, nState_, nState_);
    cMapVectorXd muState(bmuState, nState_);
    cMapMatrixXd wgtState(bwgtState, nState_, nState_);
    cMapMatrixXd varState(bvarState, nState_, nState_);
    predict(muState_pred, varState_pred,
            muState_past, varState_past,
            muState, wgtState, varState);
    return;
  }

  inline void KalmanTV::update(RefVectorXd muState_filt,
                               RefMatrixXd varState_filt,
                               cRefVectorXd& muState_pred,
                               cRefMatrixXd& varState_pred,
                               cRefVectorXd& xMeas,
                               cRefVectorXd& muMeas,
                               cRefMatrixXd& wgtMeas,
                               cRefMatrixXd& varMeas) {
    tmuMeas_.noalias() = wgtMeas * muState_pred + muMeas; // nMeas
    // std::cout << "tmuMeas_ = " << tmuMeas_ << std::endl;
    twgtMeas_.noalias() = wgtMeas * varState_pred; // nMeas x nState
    // std::cout << "twgtMeas_ = " << twgtMeas_ << std::endl;
    tvarMeas_.noalias() = twgtMeas_ * wgtMeas.adjoint() + varMeas; // nMeas x nMeas
    // std::cout << "tvarMeas_ = " << tvarMeas_ << std::endl;
    lltMeas_.compute(tvarMeas_);
    twgtMeas2_.noalias() = twgtMeas_;
    // std::cout << "twgtMeas2_ = " << twgtMeas2_ << std::endl;
    lltMeas_.solveInPlace(twgtMeas_);
    // std::cout << "twgtMeas_ = " << twgtMeas_ << std::endl;
    tmuMeas_.noalias() = xMeas - tmuMeas_;
    // std::cout << "tmuMeas_ = " << tmuMeas_ << std::endl;
    // QUESTION: is it wasteful to call adjoint() twice?
    // does it require a temporary assignment?
    muState_filt.noalias() = twgtMeas_.adjoint() * tmuMeas_ + muState_pred;
    // std::cout << "muState_filt = " << muState_filt << std::endl;
    varState_filt.noalias() = twgtMeas_.adjoint() * twgtMeas2_;
    // std::cout << "varState_filt = " << varState_filt << std::endl;
    varState_filt.noalias() = varState_pred - varState_filt;
    // std::cout << "varState_filt = " << varState_filt << std::endl;
    return;
  }
  /// Raw buffer equivalent.
  inline void KalmanTV::update(double* bmuState_filt,
                               double* bvarState_filt,
                               const double* bmuState_pred,
                               const double* bvarState_pred,
                               const double* bxMeas,
                               const double* bmuMeas,
                               const double* bwgtMeas,
                               const double* bvarMeas) {
    MapVectorXd muState_filt(bmuState_filt, nState_);
    MapMatrixXd varState_filt(bvarState_filt, nState_, nState_);
    cMapVectorXd muState_pred(bmuState_pred, nState_);
    cMapMatrixXd varState_pred(bvarState_pred, nState_, nState_);
    cMapVectorXd xMeas(bxMeas, nMeas_);
    cMapVectorXd muMeas(bmuMeas, nMeas_);
    cMapMatrixXd wgtMeas(bwgtMeas, nMeas_, nState_);
    cMapMatrixXd varMeas(bvarMeas, nMeas_, nMeas_);
    update(muState_filt, varState_filt,
           muState_pred, varState_pred,
           xMeas, muMeas, wgtMeas, varMeas);
    return;    
  }

  /// Raw buffer equivalent.
  inline void KalmanTV::filter(double* bmuState_pred,
                               double* bvarState_pred,
                               double* bmuState_filt,
                               double* bvarState_filt,
                               const double* bmuState_past,
                               const double* bvarState_past,
                               const double* bmuState,
                               const double* bwgtState,
                               const double* bvarState,
                               const double* bxMeas,
                               const double* bmuMeas,
                               const double* bwgtMeas,
                               const double* bvarMeas) {
    MapVectorXd muState_pred(bmuState_pred, nState_);
    MapMatrixXd varState_pred(bvarState_pred, nState_, nState_);
    MapVectorXd muState_filt(bmuState_filt, nState_);
    MapMatrixXd varState_filt(bvarState_filt, nState_, nState_);
    cMapVectorXd muState_past(bmuState_past, nState_);
    cMapMatrixXd varState_past(bvarState_past, nState_, nState_);
    cMapVectorXd muState(bmuState, nState_);
    cMapMatrixXd wgtState(bwgtState, nState_, nState_);
    cMapMatrixXd varState(bvarState, nState_, nState_);
    cMapVectorXd xMeas(bxMeas, nMeas_);
    cMapVectorXd muMeas(bmuMeas, nMeas_);
    cMapMatrixXd wgtMeas(bwgtMeas, nMeas_, nState_);
    cMapMatrixXd varMeas(bvarMeas, nMeas_, nMeas_);
    predict(muState_pred, varState_pred,
            muState_past, varState_past,
            muState, wgtState, varState);
    update(muState_filt, varState_filt,
           muState_pred, varState_pred,
           xMeas, muMeas, wgtMeas, varMeas);
    return;    
  }
  
  inline void KalmanTV::smooth_mv(RefVectorXd muState_smooth,
                                  RefMatrixXd varState_smooth,
                                  cRefVectorXd& muState_next,
                                  cRefMatrixXd& varState_next,
                                  cRefVectorXd& muState_filt,
                                  cRefMatrixXd& varState_filt,
                                  cRefVectorXd& muState_pred,
                                  cRefMatrixXd& varState_pred,
                                  cRefMatrixXd& wgtState) {
    tvarState_.noalias() = wgtState * varState_filt.adjoint();  
    lltState_.compute(varState_pred); 
    lltState_.solveInPlace(tvarState_); // equivalent to varState_temp_tilde
    tmuState_.noalias() = muState_next - muState_pred;
    tvarState2_.noalias() = varState_next - varState_pred;
    tvarState3_.noalias() = tvarState_.adjoint() * tvarState2_; 
    varState_smooth.noalias() = tvarState3_ * tvarState_;          
    muState_smooth.noalias() = tvarState_.adjoint() * tmuState_ + muState_filt;
    varState_smooth.noalias() = varState_smooth + varState_filt;
    return;
  }
  /// Raw buffer equivalent.
  inline void KalmanTV::smooth_mv(double* bmuState_smooth,
                                  double* bvarState_smooth,
                                  const double* bmuState_next,
                                  const double* bvarState_next,
                                  const double* bmuState_filt,
                                  const double* bvarState_filt,
                                  const double* bmuState_pred,
                                  const double* bvarState_pred,
                                  const double* bwgtState) {
    MapVectorXd muState_smooth(bmuState_smooth, nState_);
    MapMatrixXd varState_smooth(bvarState_smooth, nState_, nState_);
    cMapVectorXd muState_next(bmuState_next, nState_);
    cMapMatrixXd varState_next(bvarState_next, nState_, nState_);
    cMapVectorXd muState_filt(bmuState_filt, nState_);
    cMapMatrixXd varState_filt(bvarState_filt, nState_, nState_);
    cMapVectorXd muState_pred(bmuState_pred, nState_);
    cMapMatrixXd varState_pred(bvarState_pred, nState_, nState_);
    cMapMatrixXd wgtState(bwgtState, nState_, nState_);
    smooth_mv(muState_smooth, varState_smooth,
              muState_next, varState_next,
              muState_filt, varState_filt,
              muState_pred, varState_pred, wgtState);
    return;
  }
  
  inline void KalmanTV::smooth_sim(RefVectorXd xState_smooth,
                                   RefVectorXd muState_sim,
                                   RefMatrixXd varState_sim,
                                   cRefVectorXd& xState_next,
                                   cRefVectorXd& muState_filt,
                                   cRefMatrixXd& varState_filt,
                                   cRefVectorXd& muState_pred,
                                   cRefMatrixXd& varState_pred,
                                   cRefMatrixXd& wgtState) {
    tvarState_.noalias() = wgtState * varState_filt.adjoint();
    tvarState2_.noalias() = tvarState_; // equivalent to varState_temp
    lltState_.compute(varState_pred); 
    lltState_.solveInPlace(tvarState_); // equivalent to varState_temp_tilde
    tmuState_.noalias() = xState_next - muState_pred;
    // std::cout << "tvarState_ = " << tvarState_ << std::endl;
    tmuState_ = tvarState_.adjoint() * tmuState_ + muState_filt;
    muState_sim.noalias() = tmuState_;
    // std::cout << "tmuState_= " << tmuState_ << std::endl;
    tvarState2_ = varState_filt - tvarState_.adjoint() * tvarState2_;
    tvarState2_  = tvarState2_ * tvarState2_.adjoint();
    varState_sim.noalias() = tvarState2_; // testing
    // std::cout << "tvarState2_ =" << tvarState2_ << std::endl; 
    // Generate random draw
    std::random_device rd{};
    std::mt19937 gen{rd()};
    for (int i=0; i<nState_; ++i) {
      std::normal_distribution<double> d(0.0,1.0);
      tmuState2_(i) = d(gen);
    }
    // Cholesky
    LLT<MatrixXd> lltvarState_(tvarState2_);
    tvarState3_ = lltvarState_.matrixL();
    xState_smooth.noalias() = tmuState_ +  tvarState3_ * tmuState2_;
    return;
  }
  /// Raw buffer equivalent.
  inline void KalmanTV::smooth_sim(double* bxState_smooth,
                                   double* bmuState_sim,
                                   double* bvarState_sim,
                                   const double* bxState_next,
                                   const double* bmuState_filt,
                                   const double* bvarState_filt,
                                   const double* bmuState_pred,
                                   const double* bvarState_pred,
                                   const double* bwgtState) {
    MapVectorXd xState_smooth(bxState_smooth, nState_);
    MapVectorXd muState_sim(bmuState_sim, nState_); // for testing
    MapMatrixXd varState_sim(bvarState_sim, nState_, nState_); // for testing
    cMapVectorXd xState_next(bxState_next, nState_);
    cMapVectorXd muState_filt(bmuState_filt, nState_);
    cMapMatrixXd varState_filt(bvarState_filt, nState_, nState_);
    cMapVectorXd muState_pred(bmuState_pred, nState_);
    cMapMatrixXd varState_pred(bvarState_pred, nState_, nState_);
    cMapMatrixXd wgtState(bwgtState, nState_, nState_);
    smooth_sim(xState_smooth, muState_sim, varState_sim, 
               xState_next, muState_filt, varState_filt,
               muState_pred, varState_pred, wgtState);
    return;
  }
  
  /// Raw buffer equivalent.
  inline void KalmanTV::smooth(double* bxState_smooth,
                               double* bmuState_sim,
                               double* bvarState_sim,
                               double* bmuState_smooth,
                               double* bvarState_smooth,
                               const double* bxState_next,
                               const double* bmuState_next,
                               const double* bvarState_next,
                               const double* bmuState_filt,
                               const double* bvarState_filt,
                               const double* bmuState_pred,
                               const double* bvarState_pred,
                               const double* bwgtState) {
    MapVectorXd xState_smooth(bxState_smooth, nState_);
    MapVectorXd muState_sim(bmuState_sim, nState_); // for testing
    MapMatrixXd varState_sim(bvarState_sim, nState_, nState_); // for testing
    MapVectorXd muState_smooth(bmuState_smooth, nState_);
    MapMatrixXd varState_smooth(bvarState_smooth, nState_, nState_);
    cMapVectorXd xState_next(bxState_next, nState_);
    cMapVectorXd muState_next(bmuState_next, nState_);
    cMapMatrixXd varState_next(bvarState_next, nState_, nState_);
    cMapVectorXd muState_filt(bmuState_filt, nState_);
    cMapMatrixXd varState_filt(bvarState_filt, nState_, nState_);
    cMapVectorXd muState_pred(bmuState_pred, nState_);
    cMapMatrixXd varState_pred(bvarState_pred, nState_, nState_);
    cMapMatrixXd wgtState(bwgtState, nState_, nState_);
    smooth_mv(muState_smooth, varState_smooth,
              muState_next, varState_next,
              muState_filt, varState_filt,
              muState_pred, varState_pred, wgtState);
    smooth_sim(xState_smooth, muState_sim, varState_sim, 
               xState_next, muState_filt, varState_filt,
               muState_pred, varState_pred, wgtState);
    return;
  }                    
} // end namespace KalmanTV

// --- scratch -----------------------------------------------------------------

// class KalmanTV {
//  private:
//   int nMeas_; ///< Number of measurement dimensions.
//   int nState_; ///< Number of state dimensions.
//   int nObs_; ///< Maximum number of observations.
//   MatrixXd xMeas_; ///< Matrix of measurement vectors (columnwise).
//   MatrixXd muMeas_; ///< Matrix of measurement means.
//   MatrixXd wgtMeas_; ///< Matrix of measurement weights (columnwise concatenated).
//   MatrixXd varMeas_; ///< Matrix of measurement variances.
//   MatrixXd muState_; ///< Matrix of state means.
//   MatrixXd wgtState_; ///< Matrix of state weights (transition matrices).
//   MatrixXd varState_; ///< Matrix of state variances.
//  public:
//   /// Constructor.
//   KalmanTV(int nMeas, int nState, int nObs);
//   /// Setters and getters.
//   void set_xMeas(const Ref<const VectorXd>& x, int i);
//   void set_muMeas(const Ref<const VectorXd>& mu, int i);
//   void set_wgtMeas(const Ref<const MatrixXd>& wgt, int i);
//   void set_varMeas(const Ref<const MatrixXd>& var, int i);
//   void set_muState(const Ref<const VectorXd>& mu, int i);
//   void set_wgtState(const Ref<const MatrixXd>& wgt, int i);
//   void set_varState(const Ref<const MatrixXd>& var, int i);
//   void get_xMeas(Ref<VectorXd> x, int i);
//   void get_muMeas(Ref<VectorXd> mu, int i);
//   void get_wgtMeas(Ref<MatrixXd> wgt, int i);
//   void get_varMeas(Ref<MatrixXd> var, int i);
//   void get_muState(Ref<VectorXd> mu, int i);
//   void get_wgtState(Ref<MatrixXd> wgt, int i);
//   void get_varState(Ref<MatrixXd> var, int i);
//   /// Perform a kalman 
// };

#endif
