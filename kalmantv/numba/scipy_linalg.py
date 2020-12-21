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

# --- wrappers to BLAS/LAPACK functions ----------------------------------------

# necessary ctypes
_PTR = ctypes.POINTER
_float = ctypes.c_float
_double = ctypes.c_double
_int = ctypes.c_int
_float_p = _PTR(_float)
_double_p = _PTR(_double)
_int_p = _PTR(_int)

# dpotrf (cho_factor for doubles)
addr = get_cython_function_address('scipy.linalg.cython_lapack', 'dpotrf')
functype = ctypes.CFUNCTYPE(None,
                            _int_p,  # UPLO
                            _int_p,  # N
                            _double_p,  # U
                            _int_p,  # LDA
                            _int_p,  # INFO
                            )
_dpotrf = functype(addr)
# cho_factor: float
addr = get_cython_function_address('scipy.linalg.cython_lapack', 'spotrf')
functype = ctypes.CFUNCTYPE(None,
                            _int_p,  # UPLO
                            _int_p,  # N
                            _float_p,  # U
                            _int_p,  # LDA
                            _int_p,  # INFO
                            )
_spotrf = functype(addr)

# cho_solve: double
addr = get_cython_function_address('scipy.linalg.cython_lapack', 'dpotrs')
functype = ctypes.CFUNCTYPE(None,
                            _int_p,  # UPLO
                            _int_p,  # N
                            _int_p,  # NRHS
                            _double_p,  # U
                            _int_p,  # LDA
                            _double_p,  # X
                            _int_p,  # LDB
                            _int_p,  # INFO
                            )
_dpotrs = functype(addr)
# cho_solve: float
addr = get_cython_function_address('scipy.linalg.cython_lapack', 'spotrs')
functype = ctypes.CFUNCTYPE(None,
                            _int_p,  # UPLO
                            _int_p,  # N
                            _int_p,  # NRHS
                            _float_p,  # U
                            _int_p,  # LDA
                            _float_p,  # X
                            _int_p,  # LDB
                            _int_p,  # INFO
                            )
_spotrs = functype(addr)

# triangular matrix multiplication
# matrix-vector: double
addr = get_cython_function_address('scipy.linalg.cython_blas', 'dtrmv')
functype = ctypes.CFUNCTYPE(None,
                            _int_p,  # UPLO
                            _int_p,  # TRANS
                            _int_p,  # DIAG
                            _int_p,  # N
                            _double_p,  # A
                            _int_p,  # LDA
                            _double_p,  # X
                            _int_p,  # INCX
                            )
_dtrmv = functype(addr)
# matrix-vector: float
addr = get_cython_function_address('scipy.linalg.cython_blas', 'strmv')
functype = ctypes.CFUNCTYPE(None,
                            _int_p,  # UPLO
                            _int_p,  # TRANS
                            _int_p,  # DIAG
                            _int_p,  # N
                            _float_p,  # A
                            _int_p,  # LDA
                            _float_p,  # X
                            _int_p,  # INCX
                            )
_strmv = functype(addr)
# matrix-array: double
addr = get_cython_function_address('scipy.linalg.cython_blas', 'dtrmm')
functype = ctypes.CFUNCTYPE(None,
                            _int_p,  # SIDE
                            _int_p,  # UPLO
                            _int_p,  # TRANSA
                            _int_p,  # DIAG
                            _int_p,  # M
                            _int_p,  # N
                            _double_p,  # ALPHA
                            _double_p,  # A
                            _int_p,  # LDA
                            _double_p,  # B
                            _int_p,  # LDB
                            )
_dtrmm = functype(addr)
# matrix-array: float
addr = get_cython_function_address('scipy.linalg.cython_blas', 'strmm')
functype = ctypes.CFUNCTYPE(None,
                            _int_p,  # SIDE
                            _int_p,  # UPLO
                            _int_p,  # TRANSA
                            _int_p,  # DIAG
                            _int_p,  # M
                            _int_p,  # N
                            _float_p,  # ALPHA
                            _float_p,  # A
                            _int_p,  # LDA
                            _float_p,  # B
                            _int_p,  # LDB
                            )
_strmm = functype(addr)

# --- helper functions ---------------------------------------------------------


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
def _asfortranarray(a, overwrite_a=False):
    """
    Convert an array to fortran order, creating a copy if necessary and/or requested.
    That is, the function returns a reference to `a` if (1) `a` is in fortran order and (2) `overwrite_a = True`.  Otherwise, returns a reference to a copy of `a`, but in fortran-order.

    Note: `overwrite_a = False` seems to set `a1.flags.owndata = False` in njit but `True` in regular Python.
    """
    if not a.flags.f_contiguous:
        # copy unless a is f_contiguous
        a1 = np.asfortranarray(a)
    else:
        # np.asfortranarray creates a copy when c_contiguous is True
        a1 = a
    if (not overwrite_a) and a.flags.f_contiguous:
        # copy if a is f_contiguous and overwrite required
        a1 = np.copy(a)
    return a1

# --- njitable cho_factor -------------------------------------------------------


@overload(scipy.linalg.cho_factor)
def jit_cho_factor(a, lower=False, overwrite_a=False, check_finite=True):
    r"""
    njit implementation of `scipy.linalg.cho_factor()`.

    - As with the original scipy implementation, overwrite_a only "works" if the input array is in contiguous fortran order.  Otherwise a copy is created.

    Args:
        a (ndarray(dim1, dim2)): 2-dimensional matrix.
    
    Returns:
        (tuple): Cholesky factorization of matrix a and indicator if it is lower triangular.

    """

    # Reject non-ndarray types
    if not isinstance(a, types.Array):
        raise TypingError("Input must be a NumPy array.")
    # Reject ndarrays with non floating-point dtype
    if not isinstance(a.dtype, types.Float):
        raise TypingError("Input array must be of type float.")
    # Dimension check
    if a.ndim != 2:
        raise TypingError(
            "scipy.linalg.cho_factor() only supported on 2-D arrays.")
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
        if a.shape[0] != a.shape[1]:
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

# --- njitable cho_solve -------------------------------------------------------


@overload(scipy.linalg.cho_solve)
def jit_cho_solve(c_and_lower, b, overwrite_b=False, check_finite=True):
    r"""
    njit implementation of `scipy.linalg.cho_solve()`.

    Args:
        c_and_lower (tuple): 2-d Matrix and indicator if it is lower triangular.
        b (ndarray(dim1, dim2)): 2-d Matrix.
    
    Returns:
        (ndarray(dim1, dim2)): Solved matrix.
    """
    (c, lower) = c_and_lower
    # Reject non-ndarray types
    if not isinstance(c, types.Array) or not isinstance(b, types.Array):
        raise TypingError("c and b must be a NumPy arrays.")
    # Reject ndarrays with non floating-point dtype
    if not isinstance(c.dtype, types.Float) or not isinstance(b.dtype, types.Float):
        raise TypingError("c and b must be of type float.")
    # Dimension check
    if (c.ndim != 2) or (b.ndim > 2) or (b.ndim < 1):
        raise TypingError(
            "scipy.linalg.cho_solve() only supported on 1-D or 2-D arrays.")

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
        if (c.shape[0] != c.shape[1]):
            raise LinAlgError("The factored matrix c is not square.")
        # Dimension check
        if c.shape[1] != b.shape[0]:
            raise LinAlgError("c and b have incompatible dimensions.")
        # Check that arrays are finite
        if check_finite:
            _check_finite_array(c)
            _check_finite_array(b)
        if to_float64:
            # need to upcast these. if already float64 should be a no-copy.
            # otherwise, copies to C order, so do this before as_fortranarray
            c1 = np.asarray(c, dtype=np.float64)
            b1 = np.asarray(b, dtype=np.float64)
        else:
            c1 = c
            b1 = b
        # convert to fortran order, avoiding copy whenever possible.
        # c is assumed to come from cho_factor, so already in fortran order.
        # if it was already float64, _asfortranarray does nothing
        # if it needed to be changed to float64, a copy was already made,
        # so _asfortranarray is not overwriting the original c.
        c1 = _asfortranarray(c1, overwrite_a=True)
        b1 = _asfortranarray(b1, overwrite_b)
        # determine order
        uplo = LO if lower else UP
        # wrap everything into something with ctypes
        n = np.array(c.shape[0], dtype=np.int32)
        # nrhs = np.array(b.shape[1], dtype=np.int32)
        nrhs = np.array(b.size/b.shape[0], dtype=np.int32)
        lda = n
        ldb = n
        info = np.empty(1, dtype=np.int32)
        # call fortran via cython_lapack
        _potrs(uplo.ctypes, n.ctypes, nrhs.ctypes, c1.ctypes, lda.ctypes,
               b1.ctypes, ldb.ctypes, info.ctypes)
        # check return
        if info[0] != 0:
            raise ValueError(
                'Illegal value in internal "POTRS".'
            )
        return b1

    return cho_solve_imp

# --- njitable triangular matrix multiplication (tri_mult) ---------------------


def tri_mult(a_and_lower, x, overwrite_x=False, check_finite=True):
    r"""
    Triangular matrix multiplication :math:`a x`.

    This provides an unoptimized Python implementation for the sole purpose of testing unjitted code.  In the pure Python version, both `overwrite_x` and `check_finite` flags are ignored.
    """
    (a, lower) = a_and_lower
    # start by converting to fortran order
    a1 = np.asfortranarray(a)
    x1 = np.asfortranarray(x)
    # since a is typically coming from cho_factor
    # might have trash in the part that's not referenced.
    if lower:
        a1 = scipy.linalg.tril(a1)
    else:
        a1 = scipy.linalg.triu(a1)
    # reshape x into a matrix
    x1 = x1.reshape((x1.shape[0], -1), order="F")
    if a1.dtype != x1.dtype:
        # convert both to float64
        a1 = np.asarray(a1, dtype=np.float64)
        x1 = np.asarray(x1, dtype=np.float64)
    y = np.dot(a1, x1)
    # back to original shape
    y = y.reshape(x.shape, order="F")
    if overwrite_x:
        # inefficient, but gives the correct result
        x[:] = y
        y = x
    return y


@overload(tri_mult)
def jit_tri_mult(a_and_lower, x, overwrite_x=False, check_finite=True):
    r"""
    Triangular matrix multiplication :math:`a x`.
    
    Args:
        a_and_lower (tuple):  2-d Matrix and indicator if it is lower triangular.
        x (ndarray(dim1, dim2)): 2-d Matrix.
    
    Returns:
        (ndarray(dim1, dim2)): :math:`a x`.
    """
    (a, lower) = a_and_lower
    # Reject non-ndarray types
    if not isinstance(a, types.Array) or not isinstance(x, types.Array):
        raise TypingError("a and x must be a NumPy arrays.")
    # Reject ndarrays with non floating-point dtype
    if not isinstance(a.dtype, types.Float) or not isinstance(x.dtype, types.Float):
        raise TypingError("a and x must be of type float.")

    # type to use
    use_float64 = (a.dtype == types.float64) or (x.dtype == types.float64)
    # whether to promote to float64
    to_float64 = a.dtype != x.dtype
    # print("use_float64 = ", use_float64)
    # print("to_float64 = ", to_float64)

    # BLAS constants
    UP = np.array([ord('U')], dtype=np.int32)
    LO = np.array([ord('L')], dtype=np.int32)
    trans = np.array([ord('N')], dtype=np.int32)
    diag = np.array([ord('N')], dtype=np.int32)
    incx = np.array([1], dtype=np.int32)
    side = np.array([ord('L')], dtype=np.int32)
    alpha = np.array(1.0, dtype=np.float64 if use_float64 else np.float32)

    # Which BLAS function to use
    if x.ndim == 1:
        _trmv = _dtrmv if use_float64 else _strmv
    else:
        _trmm = _dtrmm if use_float64 else _strmm

    # implementation for matrix-vector multiplication
    def tri_mult_imp_1D(a_and_lower, x, overwrite_x=False, check_finite=True):
        (a, lower) = a_and_lower
        # Squareness check
        if (a.ndim != 2) or (a.shape[0] != a.shape[1]):
            raise LinAlgError("The triangular matrix a is not square.")
        # Dimension check
        if a.shape[1] != x.shape[0]:
            raise LinAlgError("a and x have incompatible dimensions.")
        # Check that arrays are finite
        if check_finite:
            _check_finite_array(a)
            _check_finite_array(x)
        # float and fortran conversions
        if to_float64:
            # need to upcast these. if already float64 should be a no-copy.
            # otherwise, copies to C order, so do this before as_fortranarray
            a1 = np.asarray(a, dtype=np.float64)
            x1 = np.asarray(x, dtype=np.float64)
        else:
            a1 = a
            x1 = x
        # convert to fortran order, avoiding copy whenever possible.
        a1 = _asfortranarray(a1, overwrite_a=True)
        x1 = _asfortranarray(x1, overwrite_x)
        # determine order
        uplo = LO if lower else UP
        # wrap everything into something with ctypes
        n = np.array(a.shape[0], dtype=np.int32)
        lda = n
        # call fortran via cython_blas
        _trmv(uplo.ctypes, trans.ctypes, diag.ctypes, n.ctypes,
              a1.ctypes, lda.ctypes, x1.ctypes, incx.ctypes)
        return x1

    # implementation for matrix-(array >1D) multiplication
    def tri_mult_imp_ND(a_and_lower, x, overwrite_x=False, check_finite=True):
        (a, lower) = a_and_lower
        # Squareness check
        if (a.ndim != 2) or (a.shape[0] != a.shape[1]):
            raise LinAlgError("The triangular matrix a is not square.")
        # Dimension check
        if a.shape[1] != x.shape[0]:
            raise LinAlgError("a and x have incompatible dimensions.")
        # Check that arrays are finite
        if check_finite:
            _check_finite_array(a)
            _check_finite_array(x)
        # float and fortran conversions
        if to_float64:
            # need to upcast these. if already float64 should be a no-copy.
            # otherwise, copies to C order, so do this before as_fortranarray
            a1 = np.asarray(a, dtype=np.float64)
            x1 = np.asarray(x, dtype=np.float64)
        else:
            a1 = a
            x1 = x
        # convert to fortran order, avoiding copy whenever possible.
        a1 = _asfortranarray(a1, overwrite_a=True)
        x1 = _asfortranarray(x1, overwrite_x)
        # determine order
        uplo = LO if lower else UP
        # wrap everything into something with ctypes
        m = np.array(a.shape[0], dtype=np.int32)
        n = np.array(x.size/x.shape[0], dtype=np.int32)
        lda = m
        ldb = m
        # call fortran via cython_blas
        _trmm(side.ctypes, uplo.ctypes, trans.ctypes, diag.ctypes,
              m.ctypes, n.ctypes, alpha.ctypes,
              a1.ctypes, lda.ctypes, x1.ctypes, ldb.ctypes)
        return x1

    if x.ndim == 1:
        return tri_mult_imp_1D
    else:
        return tri_mult_imp_ND
