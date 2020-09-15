# some useful scipy.linalg functions with numba overloading

import numpy as np
from numba import njit, types
from numba.extending import overload, register_jitable, \
    get_cython_function_address
from numba.core.errors import TypingError
from numpy.linalg import LinAlgError

# need this to overload properly
import scipy.linalg

# minimal wrapper to dpotrf
# from numba.extending import get_cython_function_address
# from numba import vectorize, njit
import ctypes
# import numpy as np
# from scipy.linalg.cython_lapack cimport dpotrf, dpotrs

# create the ctype functions
_PTR = ctypes.POINTER
_float = ctypes.c_float
_double = ctypes.c_double
_int = ctypes.c_int
_float_p = _PTR(_float)
_double_p = _PTR(_double)
_int_p = _PTR(_int)

# --- njittable wrappers to cho_factor and cho_solve ---------------------------
#
# LAPACK: xpotrf and xpotrs, x=d/s

# cho_factor: double
addr = get_cython_function_address('scipy.linalg.cython_lapack', 'dpotrf')
functype = ctypes.CFUNCTYPE(None,
                            _int_p,  # uplo
                            _int_p,  # n
                            _double_p,  # U
                            _int_p,  # lda
                            _int_p,  # info
                            )
_dpotrf = functype(addr)
# cho_factor: float
addr = get_cython_function_address('scipy.linalg.cython_lapack', 'spotrf')
functype = ctypes.CFUNCTYPE(None,
                            _int_p,  # uplo
                            _int_p,  # n
                            _float_p,  # U
                            _int_p,  # lda
                            _int_p,  # info
                            )
_spotrf = functype(addr)

# cho_solve: double
addr = get_cython_function_address('scipy.linalg.cython_lapack', 'dpotrs')
functype = ctypes.CFUNCTYPE(None,
                            _int_p,  # uplo
                            _int_p,  # n
                            _int_p,  # nrhs
                            _double_p,  # U
                            _int_p,  # lda
                            _double_p,  # X
                            _int_p,  # ldb
                            _int_p,  # info
                            )
_dpotrs = functype(addr)
# cho_solve: float
addr = get_cython_function_address('scipy.linalg.cython_lapack', 'spotrs')
functype = ctypes.CFUNCTYPE(None,
                            _int_p,  # uplo
                            _int_p,  # n
                            _int_p,  # nrhs
                            _float_p,  # U
                            _int_p,  # lda
                            _float_p,  # X
                            _int_p,  # ldb
                            _int_p,  # info
                            )
_spotrs = functype(addr)


@register_jitable
def _check_finite_array(a):
    """
    check whether array is finite
    (copied from numba.linalg._check_finite_matrix)
    """
    for v in np.nditer(a):
        # for v in np.ravel(a):
        if not np.isfinite(v.item()):
            raise LinAlgError("Array must not contain infs or NaNs.")


@register_jitable
def _asfortranarray(a, overwrite_a=True):
    """
    Convert an array to fortran order, creating a copy if necessary and/or requested.
    That is, the function returns a reference to `a` if (1) `a` is in fortran order and (2) `overwrite_a = True`.  Otherwise, returns a reference to a copy of `a`, but in fortran-order.

    Note: `overwrite_a = False` seems to set `a1.flags.owndata = False` in njit but `True` in regular Python.
    """
    # # this should be a no-copy if array is already in fortran order
    a1 = np.asfortranarray(a)  # copy unless a is f_contiguous
    if (not overwrite_a) and a.flags.f_contiguous:
        # copy if a is f_contiguous and overwrite required
        a1 = np.copy(a)
    return a1


@overload(scipy.linalg.cho_factor)
def jit_cho_factor(a, lower=False, overwrite_a=False, check_finite=True):
    r"""
    njit implementation of `scipy.linalg.cho_factor()`.

    - As with the original scipy implementation, overwrite_a only "works" if the input array is in contiguous fortran order.  Otherwise a copy is created.
    """

    # Reject non-ndarray types
    if not isinstance(a, types.Array):
        raise TypingError("Input must be a NumPy array.")
    # Reject ndarrays with non floating-point dtype
    if not isinstance(a.dtype, types.Float):
        raise TypingError("Input array must be of type float.")
    # # Dimension check
    # if a.ndim != 2:
    #     raise ValueError('Input array needs to be 2D but received '
    #                      'a {}d-array.'.format(a.ndim))
    # # Squareness check
    # if a.shape[0] != a.shape[1]:
    #     raise ValueError('Input array is expected to be square but has '
    #                      'the shape: {}.'.format(a.shape))

    # transpose constants
    UP = np.array([ord('U')], dtype=np.int32)
    LO = np.array([ord('L')], dtype=np.int32)

    # which LAPACK function to use
    _potrf = _dpotrf if a.dtype == types.float64 else _spotrf

    def cho_factor_imp(a, lower=False, overwrite_a=False, check_finite=True):
        # Reject ndarrays with unsupported dimensionality
        if (a.ndim != 2) or (a.shape[0] != a.shape[1]):
            raise LinAlgError('Input array needs to be a square matrix.')
        # Quick return for square empty array
        if a.size == 0:
            return a.copy(), lower
        # Check that matrix is finite
        if check_finite:
            _check_finite_array(a)
        # this should be a no-copy if array is already in fortran order
        # and overwrite_a = True
        a1 = _asfortranarray(a, overwrite_a)
        # determine order
        uplo = LO if lower else UP
        # wrap everything into something with ctypes
        n = np.array(a.shape[0], dtype=np.int32)
        lda = n
        info = np.empty(1, dtype=np.int32)
        # uplo = np.array([uplo], dtype=np.int32)
        # call fortran via cython_lapack
        _potrf(uplo.ctypes, n.ctypes, a1.ctypes,
               lda.ctypes, info.ctypes)
        # check return
        if info[0] > 0:
            # raise np.linalg.LinAlgError(
            #     "%d-th leading minor of the array is not positive "
            #     "definite." % info[0])
            raise LinAlgError("Input matrix is not positive definite.")
        if info[0] < 0:
            # info[0] = -info[0]
            # raise ValueError(
            #     'LAPACK reported an illegal value in %d-th argument'
            #     'on entry to "POTRF".' % info[0])
            raise ValueError(
                'LAPACK reported an illegal value on entry to "POTRF".'
            )
        return a1, lower

    return cho_factor_imp


@overload(scipy.linalg.cho_solve)
def jit_cho_solve(c_and_lower, b, overwrite_b=False, check_finite=True):
    r"""
    njit implementation of `scipy.linalg.cho_solve()`.
    """
    (c, lower) = c_and_lower
    # Reject non-ndarray types
    if not isinstance(c, types.Array) or not isinstance(b, types.Array):
        raise TypingError("c and b must be a NumPy arrays.")
    # Reject ndarrays with non floating-point dtype
    if not isinstance(c.dtype, types.Float) or not isinstance(b.dtype, types.Float):
        raise TypingError("c and b must be of type float.")

    # transpose constants
    UP = np.array([ord('U')], dtype=np.int32)
    LO = np.array([ord('L')], dtype=np.int32)

    # which LAPACK function to use
    _potrs = \
        _dpotrs if (c.dtype == types.float64) or (b.dtype == types.float64) \
        else _spotrs
    # whether to promote to float64
    to_float64 = c.dtype != b.dtype

    def cho_solve_imp(c_and_lower, b, overwrite_b=False, check_finite=True):
        (c, lower) = c_and_lower
        # Squareness check
        if (c.ndim != 2) or (c.shape[0] != c.shape[1]):
            raise LinAlgError("The factored matrix c is not square.")
        # Dimension check
        if c.shape[1] != b.shape[0]:
            raise LinAlgError("c and b have incompatible dimensions.")
        # Check that arrays are finite
        if check_finite:
            _check_finite_array(c)
            _check_finite_array(b)
        # c is assumed to be returned by cho_factor,
        # so no further checks are made.
        # but b needs to be fortran-ordered and/or copied.
        b1 = _asfortranarray(b, overwrite_b)
        if to_float64:
            # need to upcast these. if already float64 should be a no-copy.
            c = np.asarray(c, dtype=np.float64)
            b1 = np.asarray(b1, dtype=np.float64)
        # determine order
        uplo = LO if lower else UP
        # wrap everything into something with ctypes
        n = np.array(c.shape[0], dtype=np.int32)
        # nrhs = np.array(b[0].size, dtype=np.int32)
        nrhs = np.array(b.size/b.shape[0], dtype=np.int32)
        lda = n
        ldb = n
        info = np.empty(1, dtype=np.int32)
        # call fortran via cython_lapack
        _potrs(uplo.ctypes, n.ctypes, nrhs.ctypes, c.ctypes, lda.ctypes,
               b1.ctypes, ldb.ctypes, info.ctypes)
        # check return
        if info[0] != 0:
            raise ValueError(
                'Illegal value in internal "POTRS".'
            )
        return b1

    return cho_solve_imp
