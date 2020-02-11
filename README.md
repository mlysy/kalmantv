# **KalmanTV**: Fast Time-Varying Kalman Filtering and Smoothing

## Installation

```bash
git clone https://github.com/mlysy/kalmantv
cd kalmantv
pip install .
```

**Experimental:** Alternatively, you can install the package in "development" mode with `pip install -e .`.  This means that whenever you make changes to the package they are automatically reflected in the "installed" version.  However, I'm not sure this works for compiled code e.g., Cython, in which case it's safest to rerun `pip install .` every time you make a change.

## Unit Testing

```bash
cd kalmantv
python -m unittest discover -v
```

## Wrapping **Eigen**

- Pure C++ code: **Eigen**-based class and methods.  The methods to be wrapped in Cython must accept vectors/matrices as column-major `double*` arrays, to be converted to **Eigen** types with `Map`.  For now this is in `include/KalmanTV.h`.

- Cython wrapper: Need to convert between "memory views" and the `double*` inputs.  For now this is in `include/KalmanTV.pxd` (Cython interface) and `include/kalmantv.pyx` (Python interface).

## TODO

- [X] Add remaining **kalmantv** methods to `KalmanTV.h`, `KalmanTV.pxd`, and `kalmantv.pyx`, and of course test the methods (see how to do this in `tests/test_kalmantv.py`).

- [X] Replace test functions in `tests/test_kalmantv.py` with `kalmantv` class from **probDE**.  This is basically just copying `kalmantv.py` to `tests` folder, and adding `import kalmantv` at the top of `test_kalmantv.py`.

- [X] Test Cython interface (i.e., using `cimport`, but don't need to worry about this for now). 
