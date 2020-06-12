/// @file KalmanTV_test.cpp
/// @brief Test file for KalmanTV
///
/// Compile command:
///
/// ```
/// # with Apple Accelrate framework
/// clang++ -framework Accelerate /opt/local/lib/lapack/liblapacke.dylib -Ieigen-3.3.7 -O3 -std=c++11 -o KalmanTV_test KalmanTV_test.cpp
///
/// # with -march=native
/// clang++ -march=native -Ieigen-3.3.7 -O3 -std=c++11 -o KalmanTV_test KalmanTV_test.cpp
/// ```

// #define EIGEN_USE_BLAS
// #define EIGEN_USE_LAPACKE
#include "KalmanTV.h"
#include <chrono>
using namespace Eigen;
using namespace std::chrono;

int main() {
  Eigen::initParallel();
  
  int n_meas = 10;
  int n_state = 2000;
  int n_reps = 5;

  VectorXd mu_state_past = VectorXd::Random(n_state);
  MatrixXd var_state_past = MatrixXd::Random(n_state, n_state);
  var_state_past = var_state_past.transpose() * var_state_past;
  VectorXd mu_state = VectorXd::Random(n_state);
  MatrixXd wgt_state = MatrixXd::Random(n_state, n_state);
  MatrixXd var_state = MatrixXd::Random(n_state, n_state);
  var_state = var_state.transpose() * var_state;
  VectorXd x_meas = VectorXd::Random(n_meas);
  VectorXd mu_meas = VectorXd::Random(n_meas);
  MatrixXd wgt_meas = MatrixXd::Random(n_meas, n_state);
  MatrixXd var_meas = MatrixXd::Random(n_meas, n_meas);
  var_meas = var_meas.transpose() * var_meas;

  VectorXd mu_state_pred = VectorXd::Zero(n_state);
  MatrixXd var_state_pred = MatrixXd::Zero(n_state, n_state);
  VectorXd mu_state_filt = VectorXd::Zero(n_state);
  MatrixXd var_state_filt = MatrixXd::Zero(n_state, n_state);

  KalmanTV::KalmanTV ktv(n_meas, n_state);

  auto start = high_resolution_clock::now();
  for(int ii=0; ii<n_reps; ii++) {
    ktv.filter(mu_state_pred, var_state_pred,
	       mu_state_filt, var_state_filt,
	       mu_state_past, var_state_past,
	       mu_state, wgt_state, var_state,
	       x_meas, mu_meas, wgt_meas, var_meas);
  }
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  std::cout << "Timing: "
	    << duration.count() << " milliseconds." << std::endl;

  return 0;
}
