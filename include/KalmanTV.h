/// @file KalmanTV.h

#ifndef KalmanTV_h
#define KalmanTV_h 1

#include <Eigen/Dense>
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
    MatrixXd tvarState_; ///< Temporary storage for variance matrix.
    VectorXd tmuMeas_;
    MatrixXd twgtMeas_;
    MatrixXd twgtMeas2_;
    MatrixXd tvarMeas_;
    LLT<MatrixXd> lltMeas_;
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
    // /// Perform one step of the Kalman filter.
    // ///
    // /// Combines `predict` and `update` steps to get `theta_n|n` from `theta_n-1|n-1`.
    // void filter(RefVectorXd muState_filt,
    // 		RefMatrixXd varState_filt,
    // 		RefVectorXd muState_pred,
    // 		RefMatrixXd varState_pred,
    // 		cRefVectorXd& muState_past,
    // 		cRefMatrixXd& varState_past,
    // 		cRefVectorXd& muState,
    // 		cRefMatrixXd& wgtState,
    // 		cRefMatrixXd& varState,
    // 		cRefVectorXd& xMeas,
    // 		cRefVectorXd& muMeas,
    // 		cRefMatrixXd& wgtMeas,
    // 		cRefMatrixXd& varMeas);
    // /// Perform one step of the Kalman mean/variance smoother.
    // ///
    // /// Calculates `theta_n|N` from `theta_n+1|N`, `theta_n+1|n+1`, and `theta_n+1|n`.  **Is the indexing correct?**
    // void smooth_mv(RefVectorXd muState_smooth,
    // 		   RefMatrixXd varState_smooth,
    // 		   cRefVectorXd& muState_next,
    // 		   cRefMatrixXd& varState_next,
    // 		   cRefVectorXd& muState_filt,
    // 		   cRefMatrixXd& varState_filt,
    // 		   cRefVectorXd& muState_pred,
    // 		   cRefMatrixXd& varState_pred,
    // 		   cRefMatrixXd& wgtState);
    // /// Perform one step of the Kalman sampling smoother.
    // ///
    // /// Calculates a draw `x_n|N` from `x_n+1|N`, `theta_n+1|n+1`, and `theta_n+1|n`.  **Is the indexing correct?**
    // void smooth_sim(RefVectorXd xState_smooth,
    // 		    cRefVectorXd& xState_next,
    // 		    cRefVectorXd& muState_filt,
    // 		    cRefMatrixXd& varState_filt,
    // 		    cRefVectorXd& muState_pred,
    // 		    cRefMatrixXd& varState_pred,
    // 		    cRefMatrixXd& wgtState);
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
  };

  /// @param[in] nMeas Number of measurement variables.
  /// @param[in] nState Number of state variables.
  inline KalmanTV::KalmanTV(int nMeas, int nState) {
    // problem dimensions
    nMeas_ = nMeas;
    nState_ = nState;
    // temporary storage
    tvarState_ = MatrixXd::Identity(nState_, nState_);
    tmuMeas_ = VectorXd::Zero(nMeas_);
    tvarMeas_ = MatrixXd::Identity(nMeas_, nMeas_);
    twgtMeas_ = MatrixXd::Zero(nMeas_, nState_);
    twgtMeas2_ = MatrixXd::Zero(nMeas_, nState_);
    // cholesky solvers
    lltMeas_.compute(MatrixXd::Identity(nMeas_, nMeas_));
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
    twgtMeas2_ = twgtMeas_;
    // std::cout << "twgtMeas2_ = " << twgtMeas2_ << std::endl;
    lltMeas_.solveInPlace(twgtMeas_);
    // std::cout << "twgtMeas_ = " << twgtMeas_ << std::endl;
    tmuMeas_ = xMeas - tmuMeas_;
    // std::cout << "tmuMeas_ = " << tmuMeas_ << std::endl;
    // QUESTION: is it wasteful to call adjoint() twice?
    // does it require a temporary assignment?
    muState_filt.noalias() = twgtMeas_.adjoint() * tmuMeas_ + muState_pred;
    // std::cout << "muState_filt = " << muState_filt << std::endl;
    varState_filt.noalias() = twgtMeas_.adjoint() * twgtMeas2_;
    // std::cout << "varState_filt = " << varState_filt << std::endl;
    varState_filt = varState_pred - varState_filt;
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
