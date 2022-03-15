# kalmantv: High-Performance Kalman Filtering and Smoothing in Python

*Mohan Wu, Martin Lysy*

---

## Description

**kalmantv** provides a simple Python interface to the time-varying Kalman filtering and smoothing algorithms.  Various low-level backends are provided in the following modules:

- `kalmantv.cython`: This module performs the underlying linear algebra using the BLAS/LAPACK routines provided by NumPy through a Cython interface.  To maximize speed, no input checks are provided.  All inputs must be `float64` NumPy arrays in *Fortran* order.

- `kalmantv.eigen`: This module uses the C++ Eigen library for linear algebra.  The interface is also through Cython.  Here again we have the same input requirements and lack of checks.  Eigen is known to be faster than most BLAS/LAPACK implementations, but it needs to be compiled properly to achieve maximum performance.  In particular this involves linking against an installed version of Eigen (not provided) and setting the right compiler flags for SIMD and OpenMP support.  Some defaults are provided in `setup.py`, but tweaks may be required depending on the user's system.

- `kalmantv.numba`: This module once again uses BLAS/LAPACK but the interface is through Numba.  Here input checks are performed and the inputs can be in either C or Fortran order, and single or double precision (`float32` and `float64`).  However, C ordered arrays are first converted to Fortran order, so the latter is preferable for performance considerations.

- `kalmantv.jax`: This module uses the jit-compiled auto-diff Python library Jax. 

## Installation

```bash
git clone https://github.com/mlysy/kalmantv
cd kalmantv
pip install .
```

## Unit Testing

The unit tests are done against the **pykalman** library to ensure the same results.  From inside the root folder of the **kalmantv** source code:
```bash
pip install .[tests]
cd tests
python -m unittest discover -v
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

The usage of the library can be demonstrated through this simple example.  We use the `kalmantv` functions from `kalmantv.jax`.

Suppose we start with given mean and variance of the priror, `mu_state_past` and `var_state_past`, our goal is to compute `mu_state_filt` and `var_state_filt` at the next time step. We define `mu_state_pred` and `var_state_pred` for the prediction step in the Kalman filtering algorithm. More details can be found in the documentation.

```python
import numpy as np
from kalmantv.jax.kalmantv import *
n_meas = 2 # Set the size of the measurement
n_state = 4 # Set the size of the state

# Set initial mu and var of the prior
mu_state_past = np.random.rand(n_state) 
var_state_past = np.random.rand(n_state, n_state)
var_state_past = var_state_past.dot(var_state_past.T) #Ensure positive semidefinite
var_state_past = np.asfortranarray(var_state_past) # must use Fortran order

# Parameters to the Kalman Filter
mu_state = np.random.rand(n_state)
wgt_state = np.random.rand(n_state, n_state).T # converts to Fortran order
var_state = np.random.rand(n_state, n_state)
var_state = var_state.dot(var_state.T).T
x_meas = np.random.rand(n_meas)
mu_meas = np.random.rand(n_meas)
wgt_meas = np.random.rand(n_state, n_meas).T
var_meas = np.random.rand(n_meas, n_meas)
var_meas = var_meas.dot(var_meas.T).T

# Run the filtering algorithm
mu_state_pred, var_state_pred, mu_state_filt, var_state_filt = \
    ktv.filter(mu_state_past, var_state_past,
               mu_state, wgt_state, var_state,
               x_meas, mu_meas, wgt_meas, var_meas)
```
