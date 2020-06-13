# **kalmantv**: Fast Time-Varying Kalman Filtering and Smoothing

## Description

**kalmantv** is a Cython library that implements the Kalman Filering and Smoothing algorithms. The library consists of a low-level backend written in C++ with the state-of-the-art **Eigen** library wrapped in Cython to allow usage in Python. 

### Folder structure

**include** is pure C++ code containing the backend Kalman Filtering and Smoothing algorithms.
**kalmantv** is Cython code containing the Cython wrappers for usage in Python.

## Usage

The usage of the library can be demonstrated through this simple example.

```python
import numpy as np
from kalmantv.cython import KalmanTV
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

## Installation

```bash
git clone https://github.com/mlysy/kalmantv
cd kalmantv
pip install .
```

**Experimental:** Alternatively, you can install the package in "development" mode with `pip install -e .`.  This means that whenever you make changes to the package they are automatically reflected in the "installed" version.  However, I'm not sure this works for compiled code e.g., Cython, in which case it's safest to rerun `pip install .` every time you make a change.

## Unit Testing

The unit tests are done against the **pykalman** library to ensure the same results.
```bash
cd kalmantv
cd tests
python -m unittest discover -v
```

## Wrapping **Eigen**

- Pure C++ code: **Eigen**-based class and methods.  The methods to be wrapped in Cython must accept vectors/matrices as column-major `double*` arrays, to be converted to **Eigen** types with `Map`.  For now this is in `kalmancpp/KalmanTV.h`.

- Cython wrapper: Need to convert between "memory views" and the `double*` inputs.  For now this is in `include/KalmanTV.pxd` (Cython interface) and `include/kalmantv.pyx` (Python interface).

## TODO

- [X] Add remaining **kalmantv** methods to `KalmanTV.h`, `KalmanTV.pxd`, and `kalmantv.pyx`, and of course test the methods (see how to do this in `tests/test_kalmantv.py`).

- [X] Replace test functions in `tests/test_kalmantv.py` with `kalmantv` class from **probDE**.  This is basically just copying `kalmantv.py` to `tests` folder, and adding `import kalmantv` at the top of `test_kalmantv.py`.

- [X] Test Cython interface (i.e., using `cimport`, but don't need to worry about this for now). 
