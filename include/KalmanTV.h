/// @file KalmanTV.h

#ifndef KalmanTV_h
#define KalmanTV_h 1

// #undef NDEBUG
// #define EIGEN_RUNTIME_NO_MALLOC
#include <Eigen/Dense>
#include <iostream>

namespace KalmanTV {
  using namespace Eigen;

  /// Time-Varying Kalman Filter and Smoother.
  ///
  /// Model is
  ///
  /// ~~~~
  /// x_n = c_n + T_n x_n-1 + R_n^{1/2} eps_n   
  /// y_n = d_n + W_n x_n + H_n^{1/2} eta_n,     
  /// ~~~~
  ///
  /// where `x_n` is the state variable of dimension `s`, `y_n` is the measurement variable of dimension `m`, and `eps_n ~iid N(0, I_s)` and `eta_n ~iid N(0, I_m)` are noise variables independent of each other.  The `KalmanTV` library uses the following naming conventions to describe the variables above:
  ///
  /// - The state-level variables `x_n`, `c_n`, `T_n`, `R_n` and `eps_n` are denoted by `state`.
  /// - The measurement-level variables `y_n`, `d_n`, `W_n`, `H_n` and `eta_n` are denoted by `meas`.
  /// - The output variables `x_n` and `y_n` are denoted by `x`.
  /// - The mean vectors `c_n` and `d_n` are denoted by `mu`.
  /// - The variance matrices `R_n` and `H_n` are denoted by `var`.
  /// - The weight matrices `T_n` and `W_n` are denoted by `wgt`.
  /// - The conditional means and variances are denoted by `mu_n|m = E[x_n | y_0:m]` and `Sigma_n|m = var(x_n | y_0:m)`, and jointly as `theta_n|m = (mu_n|m, Sigma_n|m)`.
  /// - Similarly, `x_n|m` denotes a draw from `p(x_n | x_n+1, y_0:m)`.
  ///
  /// **OLD CONVENTIONS**
  ///
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
    VectorXd tmuState3_;
    MatrixXd tvarState_; ///< Temporary storage for variance matrix.
    MatrixXd tvarState2_;
    MatrixXd tvarState3_;
    MatrixXd tvarState4_;
    MatrixXd tvarState5_;
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
    /// Calculates `theta_n|n-1` from `theta_n-1|n-1`.
    void predict(RefVectorXd muState_pred,
                 RefMatrixXd varState_pred,
                 cRefVectorXd& muState_past,
                 cRefMatrixXd& varState_past,
                 cRefVectorXd& muState,
                 cRefMatrixXd& wgtState,
                 cRefMatrixXd& varState);
    /// Raw buffer equivalent.
    void predict(double* muState_pred,
                 double* varState_pred,
                 const double* muState_past,
                 const double* varState_past,
                 const double* muState,
                 const double* wgtState,
                 const double* varState);
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
    void update(double* muState_filt,
                double* varState_filt,
                const double* muState_pred,
                const double* varState_pred,
                const double* xMeas,
                const double* muMeas,
                const double* wgtMeas,
                const double* varMeas);
    /// Perform one step of the Kalman filter.
    /// Combines `predict` and `update` steps to get `theta_n|n` from `theta_n-1|n-1`.
    void filter(RefVectorXd muState_pred,
    		RefMatrixXd varState_pred,
		RefVectorXd muState_filt,
    		RefMatrixXd varState_filt,
    		cRefVectorXd& muState_past,
    		cRefMatrixXd& varState_past,
    		cRefVectorXd& muState,
    		cRefMatrixXd& wgtState,
    		cRefMatrixXd& varState,
    		cRefVectorXd& xMeas,
    		cRefVectorXd& muMeas,
    		cRefMatrixXd& wgtMeas,
    		cRefMatrixXd& varMeas);
    /// Raw buffer equivalent.
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
                const double* varMeas);
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
    void smooth_mv(double* muState_smooth,
                   double* varState_smooth,
                   const double* muState_next,
                   const double* varState_next,
                   const double* muState_filt,
                   const double* varState_filt,
                   const double* muState_pred,
                   const double* varState_pred,
                   const double* wgtState);
    /// Perform one step of the Kalman sampling smoother.
    ///
    /// Calculates a draw `x_n|N` from `x_n+1|N`, `theta_n+1|n+1`, and `theta_n+1|n`.  **Is the indexing correct?**
    void smooth_sim(RefVectorXd xState_smooth,
                    cRefVectorXd& xState_next,
                    cRefVectorXd& muState_filt,
                    cRefMatrixXd& varState_filt,
                    cRefVectorXd& muState_pred,
                    cRefMatrixXd& varState_pred,
                    cRefMatrixXd& wgtState,
                    cRefVectorXd& randState);
    /// Raw buffer equivalent.                    
    void smooth_sim(double* xState_smooth,
                    const double* xState_next,
                    const double* muState_filt,
                    const double* varState_filt,
                    const double* muState_pred,
                    const double* varState_pred,
                    const double* wgtState,
                    const double* randState);    
    /// Perfrom one step of both Kalman mean/variance and sampling smoothers.
    void smooth(RefVectorXd xState_smooth,
                RefVectorXd muState_smooth,
                RefMatrixXd varState_smooth,
                cRefVectorXd& xState_next,
                cRefVectorXd& muState_next,
                cRefMatrixXd& varState_next,
                cRefVectorXd& muState_filt,
                cRefMatrixXd& varState_filt,
                cRefVectorXd& muState_pred,
                cRefMatrixXd& varState_pred,
                cRefMatrixXd& wgtState,
                cRefVectorXd& randState);
    /// Raw buffer equivalent.                    
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
                const double* randState); 
    // void printX() {
    //   int n = 2;
    //   int p = 3;
    //   MatrixXd X = MatrixXd::Constant(n, p, 3.14);
    //   std::cout << "X =\n" << X << std::endl;
    //   return;
    // }
  };

  /// @param[in] nMeas Number of measurement variables.
  /// @param[in] nState Number of state variables.
  inline KalmanTV::KalmanTV(int nMeas, int nState) {
    // Eigen::internal::set_is_malloc_allowed(true);
    // problem dimensions
    nMeas_ = nMeas;
    nState_ = nState;
    // temporary storage
    tmuState_ = VectorXd::Zero(nState_);
    tmuState2_ = VectorXd::Zero(nState_);
    // tmuState3_ = VectorXd::Zero(nState_);
    tvarState_ = MatrixXd::Identity(nState_, nState_);
    tvarState2_ = MatrixXd::Identity(nState_, nState_);
    tvarState3_ = MatrixXd::Identity(nState_, nState_);
    tvarState4_ = MatrixXd::Identity(nState_, nState_);
    tvarState5_ = MatrixXd::Identity(nState_, nState_);
    tmuMeas_ = VectorXd::Zero(nMeas_);
    tvarMeas_ = MatrixXd::Identity(nMeas_, nMeas_);
    twgtMeas_ = MatrixXd::Zero(nMeas_, nState_);
    twgtMeas2_ = MatrixXd::Zero(nMeas_, nState_);
    // cholesky solvers
    lltMeas_.compute(MatrixXd::Identity(nMeas_, nMeas_));
    lltState_.compute(MatrixXd::Identity(nState_, nState_));
    // Eigen::internal::set_is_malloc_allowed(false);
  }

  /// @param[out] muState_pred Predicted state mean `mu_n|n-1`.
  /// @param[out] varState_pred Predicted state variance `Sigma_n|n-1`.
  /// @param[in] muState_past Previous state mean `mu_n-1|n-1`.
  /// @param[in] varState_past Previous state variance `Sigma_n-1|n-1`.
  /// @param[in] muState Current state mean `c_n`.
  /// @param[in] wgtState Current state transition matrix `T_n`.
  /// @param[in] varState Current state variance `R_n`.
  inline void KalmanTV::predict(RefVectorXd muState_pred,
                                RefMatrixXd varState_pred,
                                cRefVectorXd& muState_past,
                                cRefMatrixXd& varState_past,
                                cRefVectorXd& muState,
                                cRefMatrixXd& wgtState,
                                cRefMatrixXd& varState) {
    muState_pred.noalias() = wgtState * muState_past;
    muState_pred += muState;
    // // need to assign to temporary for matrix triple product
    tvarState_.noalias() = wgtState * varState_past;
    varState_pred.noalias() = tvarState_ * wgtState.adjoint();
    varState_pred += varState;
    return;
  }
  /// @note Arguments updated to be identical to those with `Eigen` types, so we don't need to re-document.
  inline void KalmanTV::predict(double* muState_pred,
                                double* varState_pred,
                                const double* muState_past,
                                const double* varState_past,
                                const double* muState,
                                const double* wgtState,
                                const double* varState) {
    MapVectorXd muState_pred_(muState_pred, nState_);
    MapMatrixXd varState_pred_(varState_pred, nState_, nState_);
    cMapVectorXd muState_past_(muState_past, nState_);
    cMapMatrixXd varState_past_(varState_past, nState_, nState_);
    cMapVectorXd muState_(muState, nState_);
    cMapMatrixXd wgtState_(wgtState, nState_, nState_);
    cMapMatrixXd varState_(varState, nState_, nState_);
    predict(muState_pred_, varState_pred_,
            muState_past_, varState_past_,
            muState_, wgtState_, varState_);
    return;
  }

  /// @param[out] muState_filt Current state mean `mu_n|n`.
  /// @param[out] varState_filt Current state variance `Sigma_n|n`.
  /// @param[in] muState_pred Predicted state mean `mu_n|n-1`.
  /// @param[in] varState_pred Predicted state variance `Sigma_n|n-1`.
  /// @param[in] xMeas Current measure `y_n`.
  /// @param[in] muMeas Current measure mean `d_n`.
  /// @param[in] wgtMeas Current measure transition matrix `W_n`.
  /// @param[in] varMeas Current measure variance `H_n`.
  inline void KalmanTV::update(RefVectorXd muState_filt,
                               RefMatrixXd varState_filt,
                               cRefVectorXd& muState_pred,
                               cRefMatrixXd& varState_pred,
                               cRefVectorXd& xMeas,
                               cRefVectorXd& muMeas,
                               cRefMatrixXd& wgtMeas,
                               cRefMatrixXd& varMeas) {
    tmuMeas_.noalias() = wgtMeas * muState_pred;
    tmuMeas_ += muMeas; // nMeas
    // std::cout << "tmuMeas_ = " << tmuMeas_ << std::endl;
    twgtMeas_.noalias() = wgtMeas * varState_pred; // nMeas x nState
    // std::cout << "twgtMeas_ = " << twgtMeas_ << std::endl;
    tvarMeas_.noalias() = twgtMeas_ * wgtMeas.adjoint();
    tvarMeas_ += varMeas; // nMeas x nMeas
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
    muState_filt.noalias() = twgtMeas_.adjoint() * tmuMeas_ ;
    muState_filt += muState_pred;
    // std::cout << "muState_filt = " << muState_filt << std::endl;
    varState_filt.noalias() = twgtMeas_.adjoint() * twgtMeas2_;
    // std::cout << "varState_filt = " << varState_filt << std::endl;
    varState_filt.noalias() = varState_pred - varState_filt;
    // std::cout << "varState_filt = " << varState_filt << std::endl;
    return;
  }
  /// Raw buffer equivalent.
  inline void KalmanTV::update(double* muState_filt,
                               double* varState_filt,
                               const double* muState_pred,
                               const double* varState_pred,
                               const double* xMeas,
                               const double* muMeas,
                               const double* wgtMeas,
                               const double* varMeas) {
    MapVectorXd muState_filt_(muState_filt, nState_);
    MapMatrixXd varState_filt_(varState_filt, nState_, nState_);
    cMapVectorXd muState_pred_(muState_pred, nState_);
    cMapMatrixXd varState_pred_(varState_pred, nState_, nState_);
    cMapVectorXd xMeas_(xMeas, nMeas_);
    cMapVectorXd muMeas_(muMeas, nMeas_);
    cMapMatrixXd wgtMeas_(wgtMeas, nMeas_, nState_);
    cMapMatrixXd varMeas_(varMeas, nMeas_, nMeas_);
    update(muState_filt_, varState_filt_,
           muState_pred_, varState_pred_,
           xMeas_, muMeas_, wgtMeas_, varMeas_);
    return;    
  }
  
  /// @param[out] muState_pred Predicted state mean `mu_n|n-1`.
  /// @param[out] varState_pred Predicted state variance `Sigma_n|n-1`.
  /// @param[out] muState_filt Current state mean `mu_n|n`.
  /// @param[out] varState_filt Current state variance `Sigma_n|n`.
  /// @param[in] muState_past Previous state mean `mu_n-1|n-1`.
  /// @param[in] varState_past Previous state variance `Sigma_n-1|n-1`.
  /// @param[in] muState Current state mean `c_n`.
  /// @param[in] wgtState Current state transition matrix `T_n`.
  /// @param[in] varState Current state variance `R_n`.
  /// @param[in] xMeas Current measure `y_n`.
  /// @param[in] muMeas Current measure mean `d_n`.
  /// @param[in] wgtMeas Current measure transition matrix `W_n`.
  /// @param[in] varMeas Current measure variance `H_n`.
  inline void KalmanTV::filter(RefVectorXd muState_pred,
			       RefMatrixXd varState_pred,
			       RefVectorXd muState_filt,
			       RefMatrixXd varState_filt,
			       cRefVectorXd& muState_past,
			       cRefMatrixXd& varState_past,
			       cRefVectorXd& muState,
			       cRefMatrixXd& wgtState,
			       cRefMatrixXd& varState,
			       cRefVectorXd& xMeas,
			       cRefVectorXd& muMeas,
			       cRefMatrixXd& wgtMeas,
			       cRefMatrixXd& varMeas) {
    predict(muState_pred, varState_pred,
            muState_past, varState_past,
            muState, wgtState, varState);
    update(muState_filt, varState_filt,
           muState_pred, varState_pred,
           xMeas, muMeas, wgtMeas, varMeas);
    return;    
  }
  /// Raw buffer equivalent.
  inline void KalmanTV::filter(double* muState_pred,
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
                               const double* varMeas) {
    predict(muState_pred, varState_pred,
            muState_past, varState_past,
            muState, wgtState, varState);
    update(muState_filt, varState_filt,
           muState_pred, varState_pred,
           xMeas, muMeas, wgtMeas, varMeas);
    return;    
  }
  
  /// @param[out] muState_smooth Smoothed state mean `mu_n|N`.
  /// @param[out] varState_smooth Smoothed state variance `Sigma_n|N`.
  /// @param[in] muState_next Next smoothed state mean `mu_n+1|N`.
  /// @param[in] varState_next Next smoothed state variance `Sigma_n+1|N`.
  /// @param[in] muState_filt Current state mean `mu_n|n`.
  /// @param[in] varState_filt Current state variance `Sigma_n|n`.
  /// @param[in] muState_pred Predicted state mean `mu_n+1|n`.
  /// @param[in] varState_pred Predicted state variance `Sigma_n+1|n`.
  /// @param[in] wgtMeas Current measure transition matrix `W_n`.
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
    muState_smooth.noalias() = tvarState_.adjoint() * tmuState_;
    muState_smooth += muState_filt;
    varState_smooth += varState_filt;
    return;
  }
  /// Raw buffer equivalent.
  inline void KalmanTV::smooth_mv(double* muState_smooth,
                                  double* varState_smooth,
                                  const double* muState_next,
                                  const double* varState_next,
                                  const double* muState_filt,
                                  const double* varState_filt,
                                  const double* muState_pred,
                                  const double* varState_pred,
                                  const double* wgtState) {
    MapVectorXd muState_smooth_(muState_smooth, nState_);
    MapMatrixXd varState_smooth_(varState_smooth, nState_, nState_);
    cMapVectorXd muState_next_(muState_next, nState_);
    cMapMatrixXd varState_next_(varState_next, nState_, nState_);
    cMapVectorXd muState_filt_(muState_filt, nState_);
    cMapMatrixXd varState_filt_(varState_filt, nState_, nState_);
    cMapVectorXd muState_pred_(muState_pred, nState_);
    cMapMatrixXd varState_pred_(varState_pred, nState_, nState_);
    cMapMatrixXd wgtState_(wgtState, nState_, nState_);
    smooth_mv(muState_smooth_, varState_smooth_,
              muState_next_, varState_next_,
              muState_filt_, varState_filt_,
              muState_pred_, varState_pred_, wgtState_);
    return;
  }
  
  /// @param[out] xState_smooth Smoothed state `X_n`.
  /// @param[in] muState_next Next smoothed state mean `mu_n+1|N`.
  /// @param[in] varState_next Next smoothed state variance `Sigma_n+1|N`.
  /// @param[in] muState_filt Current state mean `mu_n|n`.
  /// @param[in] varState_filt Current state variance `Sigma_n|n`.
  /// @param[in] muState_pred Predicted state mean `mu_n+1|n`.
  /// @param[in] varState_pred Predicted state variance `Sigma_n+1|n`.
  /// @param[in] wgtMeas Current measure transition matrix `W_n`.
  /// @param[in] randState Random draws from `N(0,1)` for simulating the smoothed state.
  inline void KalmanTV::smooth_sim(RefVectorXd xState_smooth,
                                   cRefVectorXd& xState_next,
                                   cRefVectorXd& muState_filt,
                                   cRefMatrixXd& varState_filt,
                                   cRefVectorXd& muState_pred,
                                   cRefMatrixXd& varState_pred,
                                   cRefMatrixXd& wgtState,
                                   cRefVectorXd& randState) {
    tvarState_.noalias() = wgtState * varState_filt.adjoint();
    tvarState2_.noalias() = tvarState_; // equivalent to varState_temp
    lltState_.compute(varState_pred); 
    lltState_.solveInPlace(tvarState_); // equivalent to varState_temp_tilde
    tmuState_.noalias() = xState_next - muState_pred;
    // std::cout << "tvarState_ = " << tvarState_ << std::endl;
    tmuState2_.noalias() = tvarState_.adjoint() * tmuState_;
    tmuState2_ += muState_filt;
    // muState_sim.noalias() = tmuState2_;
    // std::cout << "tmuState_= " << tmuState_ << std::endl;
    tvarState3_.noalias() = tvarState_.adjoint() * tvarState2_;
    tvarState3_.noalias() = varState_filt - tvarState3_;
    tvarState4_.noalias() = tvarState3_ * tvarState3_.adjoint(); // only for testing (requires semi-positive)
    // varState_sim.noalias() = tvarState4_; // testing
    // Cholesky
    lltState_.compute(tvarState4_); // use tvarState3_ in the algorithm
    tvarState5_ = lltState_.matrixL();
    xState_smooth.noalias() = tvarState5_ * randState;
    xState_smooth += tmuState2_;
    return;
  }
  /// Raw buffer equivalent.
  inline void KalmanTV::smooth_sim(double* xState_smooth,
                                   const double* xState_next,
                                   const double* muState_filt,
                                   const double* varState_filt,
                                   const double* muState_pred,
                                   const double* varState_pred,
                                   const double* wgtState,
                                   const double* randState) {
    MapVectorXd xState_smooth_(xState_smooth, nState_);
    cMapVectorXd xState_next_(xState_next, nState_);
    cMapVectorXd muState_filt_(muState_filt, nState_);
    cMapMatrixXd varState_filt_(varState_filt, nState_, nState_);
    cMapVectorXd muState_pred_(muState_pred, nState_);
    cMapMatrixXd varState_pred_(varState_pred, nState_, nState_);
    cMapMatrixXd wgtState_(wgtState, nState_, nState_);
    cMapVectorXd randState_(randState, nState_);
    smooth_sim(xState_smooth_, xState_next_, 
               muState_filt_, varState_filt_,
               muState_pred_, varState_pred_, 
               wgtState_, randState_);
    return;
  }

  /// @param[out] xState_smooth Smoothed state `X_n`.
  /// @param[out] muState_smooth Smoothed state mean `mu_n|N`.
  /// @param[out] varState_smooth Smoothed state variance `Sigma_n|N`.
  /// @param[in] muState_next Next smoothed state mean `mu_n+1|N`.
  /// @param[in] varState_next Next smoothed state variance `Sigma_n+1|N`.
  /// @param[in] muState_filt Current state mean `mu_n|n`.
  /// @param[in] varState_filt Current state variance `Sigma_n|n`.
  /// @param[in] muState_pred Predicted state mean `mu_n+1|n`.
  /// @param[in] varState_pred Predicted state variance `Sigma_n+1|n`.
  /// @param[in] wgtMeas Current measure transition matrix `W_n`.
  /// @param[in] randState Random draws from `N(0,1)` for simulating the smoothed state.
  inline void KalmanTV::smooth(RefVectorXd xState_smooth,
                               RefVectorXd muState_smooth,
                               RefMatrixXd varState_smooth,
                               cRefVectorXd& xState_next,
                               cRefVectorXd& muState_next,
                               cRefMatrixXd& varState_next,
                               cRefVectorXd& muState_filt,
                               cRefMatrixXd& varState_filt,
                               cRefVectorXd& muState_pred,
                               cRefMatrixXd& varState_pred,
                               cRefMatrixXd& wgtState,
                               cRefVectorXd& randState) {
    smooth_mv(muState_smooth, varState_smooth,
              muState_next, varState_next,
              muState_filt, varState_filt,
              muState_pred, varState_pred, wgtState);
    smooth_sim(xState_smooth,xState_next, 
               muState_filt, varState_filt,
               muState_pred, varState_pred, 
               wgtState, randState);
    return;
  }
  /// Raw buffer equivalent.
  inline void KalmanTV::smooth(double* xState_smooth,
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
                               const double* randState) {
    smooth_mv(muState_smooth, varState_smooth,
              muState_next, varState_next,
              muState_filt, varState_filt,
              muState_pred, varState_pred, wgtState);
    smooth_sim(xState_smooth, xState_next, 
               muState_filt, varState_filt,
               muState_pred, varState_pred, 
               wgtState, randState);
    return;
  }                    
} // end namespace KalmanTV


#endif
