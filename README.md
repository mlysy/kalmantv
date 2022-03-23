# kalmantv: High-Performance Kalman Filtering and Smoothing in Python

*Mohan Wu, Martin Lysy*

---

## Description

**kalmantv** provides a simple Python interface to the time-varying Kalman filtering and smoothing algorithms.  The underlying model is

*x_n = Q_n (x_{n-1} -lambda_n) + lambda_n + R_n^{1/2} eps_n*

*y_n = d_n + W x_n + Sigma_n^{1/2} eta_n*,

where *eps_n* and *eta_n* are independent vectors of iid standard normals of size `n_state` and `n_meas`, respectively.  The Kalman filtering and smoothing algorithms are efficient ways of calculating

- *E[x_n | y_{0:n}]* and *var(x_n | y_{0:n})* (filtering).

- *E[x_n | y_{0:N}]* and *var(x_n | y_{0:N})* (smoothing).

- *E[x_n | x_{n+1}, y_{0:N}]* and *var(x_n | x_{n+1}, y_{0:N})* (smoothed sampling).

Various low-level backends are provided in the following modules:

- `kalmantv.cython`: This module performs the underlying linear algebra using the BLAS/LAPACK routines provided by NumPy through a Cython interface.  

    **Warning:** To improve performance, all NumPy matrices must be `float64` arrays *supplied in Fortran order*.  No checks are made internally for this, i.e., supplying matrices in C order will produce incorrect results.


- `kalmantv.eigen`: This module uses the C++ Eigen library for linear algebra.  The interface is also through Cython.  Here again we have the same input requirements and lack of checks.  Eigen is known to be faster than most BLAS/LAPACK implementations, but it needs to be carefully compiled to achieve maximum performance.  In particular, this involves linking against an up-to-date version of Eigen (provide by the Python package **eigenpip**) and setting the right compiler flags for SIMD and OpenMP support.  Some defaults are provided in `setup.py`, but tweaks may be required depending on the user's system.

- `kalmantv.numba`: This module once again uses BLAS/LAPACK for linear algebra, but the high-level interface is through Numba.  Here input checks are performed and the inputs can be in either Fortran or C order, and of single or double precision (`float32` and `float64`).  However, C ordered arrays are first converted to Fortran order, so for performance considerations the latter is preferable.


## Installation

For the PyPi version, simply do `pip install .`.  For the latest (stable) development version:

```bash
git clone https://github.com/mlysy/kalmantv
cd kalmantv
pip install .
```

## Unit Testing

The unit tests are done against the **pykalman** library to ensure the same results.  From inside the root folder of the **kalmantv** source code:
```bash
pip install .[tests]
tox
```

## Documentation

The HTML documentation can be compiled from the **kalmantv** root folder:
```bash
pip install .[docs]
cd docs
make html
```
This will create the documentation in `docs/build`.

## Usage

The following example illustrates how to run one step of the Kalman filtering algorithm.  This is done using the `filter()` method of the `KalmanTV` class in the `kalmantv.cython` module.  The same class is defined in `kalmantv.eigen` and `kalmantv.numba` modules with exactly the same methods and signatures.

Starting with *E[x_{n-1} | y_{0:n-1}]* and *var(x_{n-1} | y_{0:n-1})* obtained at time *n-1*, the goal is to compute *E[x_{n-1} | y_{0:n-1}]* and *var(x_{n-1} | y_{0:n-1})* at the next time step *n*.

```python
import numpy as np
from kalmantv.cython import KalmanTV
n_meas = 2 # Set the size of the measurement
n_state = 4 # Set the size of the state

# Initialize mean and variance at tim n-1
mu_state_past = np.random.rand(n_state)                # E[x_{n-1} | y_{0:n-1}]
var_state_past = np.random.rand(n_state, n_state)      # var(x_{n-1} | y_{0:n-1})
var_state_past = var_state_past.dot(var_state_past.T)  # ensure symmetric positive definiteness
var_state_past = np.asfortranarray(var_state_past)     # convert to Fortran order

# Parameters to the Kalman Filter
mu_state = np.random.rand(n_state)               # lambda_n
wgt_state = np.random.rand(n_state, n_state).T   # Q_n
var_state = np.random.rand(n_state, n_state)     # R_n
var_state = var_state.dot(var_state.T).T
x_meas = np.random.rand(n_meas)                  # y_n
mu_meas = np.random.rand(n_meas)                 # d_n
wgt_meas = np.random.rand(n_state, n_meas).T     # W_n
var_meas = np.random.rand(n_meas, n_meas)        # Sigma_n
var_meas = var_meas.dot(var_meas.T).T

# Initialize the KalmanTV class
ktv = KalmanTV(n_meas, n_state)

# Allocate memory for storing the output
mu_state_pred = np.empty(n_state)                         # E[x_n | y_{0:n-1}]
var_state_pred = np.empty((n_state, n_state), order='F')  # var(x_n | y_{0:n-1})
mu_state_filt = np.empty(n_state)                         # E[x_n | y_{0:n}]
var_state_filt = np.empty((n_state, n_state), order='F')  # var(x_n | y_{0:n})

# Run the filtering algorithm
ktv.filter(mu_state_pred, var_state_pred,
           mu_state_filt, var_state_filt,
           mu_state_past, var_state_past,
           mu_state, wgt_state, var_state,
           x_meas, mu_meas, wgt_meas, var_meas)
```

A similar interface for Kalman smoothing and smoothed sampling is provided by the methods `KalmanTV.smooth_mv()` and `KalmanTV.smooth_sim()`.  Please see documentation for details.
