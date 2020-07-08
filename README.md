# kalmantv: Fast Time-Varying Kalman Filtering and Smoothing

*Mohan Wu, Martin Lysy*

---

## Description

**kalmantv** provides a simple interface to the time-varying Kalman filtering and smoothing algorithms.  The backend is written in Cython which gives a considerable performance boost for small and moderate problems compared to a pure Python implementation.  Also provided are calculations for general Gaussian state-space models which don't use the Kalman recursions.  These calculations can be used to test future implementations of the Kalman filter and smoother for more general problems.

## Installation

```bash
git clone https://github.com/mlysy/kalmantv
cd kalmantv
pip install .
```

## Unit Testing

The unit tests are done against the **pykalman** library to ensure the same results.  From inside the root folder of the **kalmantv** source code:
```bash
cd tests
python -m unittest discover -v
```

## Documentation

The HTML documentation can be compiled from the **kalmantv** root folder:
```bash
cd docs
make html
```
This will create the documentation in `docs/build`.

## Usage

The usage of the library can be demonstrated through this simple example.

```python
import numpy as np
from kalmantv import KalmanTV
n_meas = 2 # Set the size of the measurement
n_state = 4 # Set the size of the state

# Set initial mu and var of the prior
mu_state_past = np.random.rand(n_state) 
var_state_past = np.random.rand(n_state, n_state)
var_state_past = var_state_past.dot(var_state_past.T) #Ensure positive semidefinite

# Parameters to the Kalman Filter
mu_state = np.random.rand(n_state)
wgt_state = np.random.rand(n_state, n_state)
var_state = np.random.rand(n_state, n_state)
var_state = var_state.dot(var_state.T)
x_meas = np.random.rand(n_meas)
mu_meas = np.random.rand(n_meas)
wgt_meas = np.random.rand(n_meas, n_state)
var_meas = np.random.rand(n_meas, n_meas)
var_meas = var_meas.dot(var_meas.T)

# Initialize the KalmanTV class
ktv = KalmanTV(n_meas, n_state)

# Allocate memory for storing the output
mu_state_pred = np.empty(n_state)
var_state_pred = np.empty((n_state, n_state), order='F')
mu_state_filt = np.empty(n_state)
var_state_filt = np.empty((n_state, n_state), order='F')

# Run the filtering algorithm
ktv.filter(mu_state_pred, var_state_pred,
	       mu_state_filt, var_state_filt,
           mu_state_past, var_state_past,
           mu_state, wgt_state, var_state,
           x_meas, mu_meas, wgt_meas, var_meas)
```

## TODO

- [ ] Naming conventions: Plural of argument names, maybe something more memorable than `gss`?
