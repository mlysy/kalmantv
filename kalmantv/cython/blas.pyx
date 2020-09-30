# cython: boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False
import numpy as np
cimport numpy as np
cimport cython
cimport scipy.linalg.cython_blas as blas
from scipy.linalg.cython_lapack cimport dpotrf, dpotrs, dlacpy

cpdef void vec_copy(double[::1] y,
                    const double[::1] x):
    r"""
    Copies vector x to y.

    Args:
        y (ndarray(N)): Returned vector.
        x (ndarray(N)): Vector x.

    Returns:
        (ndarray(M)): Copied vector y.

    """
    cdef int N = len(x), incx = 1, incy = 1
    blas.dcopy( & N, & x[0], & incx, & y[0], & incy)
    return

cpdef void mat_copy(double[::1, :] B,
                    const double[::1, :] A):
    r"""
    Copies Matrix A to B.

    Args:
        B (ndarray(N)): Returned matrix.
        A (ndarray(N)): Matrix A.

    Returns:
        (ndarray(M)): Copied matrix B.

    """
    cdef char * uplo = 'A'
    cdef int M = A.shape[0], N = A.shape[1], lda = M, ldb = M
    dlacpy( & uplo[0], & M, & N, & A[0, 0], & lda, & B[0, 0], & ldb)
    return

cpdef void vec_add(double[::1] y,
                   const double alpha,
                   const double[::1] x):
    r"""
    Calculates :math:`y = \alpha x + y`.

    Args:
        y (ndarray(N)): Returned vector.
        alpha (double): Scalar :math:`\alpha`.
        x (ndarray(N)): Vector x.

    Returns:
        (ndarray(M)): :math:`y = \alpha x + y`.

    """
    cdef int N = len(x), incx = 1, incy = 1
    blas.daxpy(& N, & alpha, & x[0], & incx, & y[0], & incy)
    return

cpdef void mat_add(double[::1, :] B,
                   const double alpha,
                   const double beta,
                   const double[::1, :] A):
    r"""
    Calculates :math:`B = \alpha A + \beta B`.

    Args:
        B (ndarray(N)): Returned matrix.
        alpha (double): Scalar :math:`\alpha`.
        beta (double): Scalar :math:`\beta`.
        A (ndarray(N)): Matrix A.
        
    Returns:
        (ndarray(M)): :math:`B = \alpha A + \beta B`.

    """
    cdef int M = A.shape[0], N = A.shape[1]
    for i in range(M):
        for j in range(N):
            B[i, j] = alpha*A[i, j] + beta*B[i, j]
    return

cpdef void mat_vec_mult(double[::1] y,
                        char * trans,
                        const double alpha,
                        const double beta,
                        const double[::1, :] A,
                        const double[::1] x):
    r"""
    Calculates :math:`y = \alpha A x + \beta y`.

    Args:
        y (ndarray(M)): Returned vector.
        trans (char): Specifies if matrix A should be transposed.
        alpha (double): Scalar :math:`\alpha`.
        beta (double): Scalar :math:`\beta`.
        A (ndarray(M, N)): Matrix A.
        x (ndarray(N)): Vector x.
        
    Returns:
        (ndarray(M)): :math:`y = \alpha A x + \beta y`.

    """
    cdef int M = A.shape[0], N = A.shape[1], lda = M, incx = 1, incy = 1
    blas.dgemv( & trans[0], & M, & N, & alpha, & A[0, 0], & lda, & x[0], & incx, & beta, & y[0], & incy)
    return

cpdef void tri_vec_mult(double[::1] x,
                        char * uplo,
                        char * trans,
                        char * diag,
                        const double[::1, :] A):
    r"""
    Calculates :math:`x = A x` where A is a triangular matrix.

    Args:
        x (ndarray(N)): Vector x.
        uplo (char): Specifies if matrix A is upper or lower triangular.
        trans (char): Specifies if matrix A should be transposed.
        diag (char): Specifies if matrix A is unit triangular.
        A (ndarray(M, N)): Matrix A.

    Returns:
        (ndarray(M)): :math:`x = A x`.

    """
    cdef int N = A.shape[0], lda = N, incx = 1
    blas.dtrmv(& uplo[0], & trans[0], & diag[0], & N, & A[0, 0], & lda, & x[0], & incx)
    return

cpdef void mat_mult(double[::1, :] C,
                    char * transa,
                    char * transb,
                    const double alpha,
                    const double beta,
                    const double[::1, :] A,
                    const double[::1, :] B):
    r"""
    Calculates :math:`C = \alpha A B + \beta C`.

    Args:
        C (ndarray(M, N)): Returned matrix.
        transa (char): Specifies if matrix A should be transposed.
        transb (char): Specifies if matrix B should be transposed.
        alpha (double): Scalar :math:`\alpha`.
        beta (double): Scalar :math:`\beta`.
        A (ndarray(M, K)): First matrix.
        B (ndarray(K, N)): Second matrix.

    Returns:
        (ndarray(M, N)): :math:`C = \alpha A B + \beta C`.

    """
    # get dimensions
    cdef int M, N, K, lda, ldb, ldc
    if transa == b'N':
        M = A.shape[0]
        K = A.shape[1]
        lda = M
    else:
        M = A.shape[1]
        K = A.shape[0]
        lda = K

    if transb == b'N':
        N = B.shape[1]
        ldb = K
    else:
        N = B.shape[0]
        ldb = N
    ldc = M
    blas.dgemm( & transa[0], & transb[0], & M, & N, & K, & alpha, & A[0, 0], & lda, & B[0, 0], & ldb, & beta, & C[0, 0], & ldc)
    return

cpdef void mat_triple_mult(double[::1, :] D,
                           double[::1, :] temp,
                           char * transa,
                           char * transb,
                           char * transc,
                           const double alpha,
                           const double beta,
                           const double[::1, :] A,
                           const double[::1, :] B,
                           const double[::1, :] C):
    r"""
    Calculates :math:`D = \alpha A B C + \beta D`.

    Args:
        D (ndarray(L, N)): Returned matrix.
        temp (ndarray(M, L)): Temp matrix for intermediate matrix multiplication; :math:`AB`.
        transa (char): Specifies if matrix A should be transposed.
        transb (char): Specifies if matrix B should be transposed.
        transc (char): Specifies if matrix C should be transposed.
        alpha (double): Scalar :math:`\alpha`.
        beta (double): Scalar :math:`\beta`.
        A (ndarray(M, K)): First matrix.
        B (ndarray(K, L)): Second matrix.
        C (ndarray(L, N)): Third matrix.

    Returns:
        (tuple):
        - **D** (ndarray(M, N)): :math:`D = \alpha A B C + \beta D`.
        - **temp** (ndarray(M, L)): :math:`AB`.

    """
    # Temp alpha, beta
    cdef int alpha1 = 1, beta1 = 0
    # Temp trans
    cdef char * trans1 = 'N'
    mat_mult(temp, transa, transb, alpha1, beta1, A, B)
    mat_mult(D, trans1, transc, alpha, beta, temp, C)
    return

cpdef void chol_fact(double[::1, :] L,
                     const double[::1, :] V):
    r"""
    Computes the cholesky factorization of variance matrix V.

    Args:
        L (ndarray(N, M)): Returned matrix.
        V (ndarray(N, N)): Variance matrix.
    
    Returns:
        (ndarray(N, M)): Cholesky factorization of variance matrix V.

    """
    cdef char * uplo = 'L'
    cdef int N = V.shape[0], lda = N, info
    mat_copy(L, V)  # operates in-place, so this prevents V from being overwritten
    dpotrf(& uplo[0], & N, & L[0, 0], & lda, & info)
    return

cpdef void solveV(double[::1, :] U,
                  double[::1, :] X,
                  const double[::1, :] V,
                  const double[::1, :] B):
    r"""
    Solves X in :math:`VX = B`, where V is a variance matrix.

    Args:
        U (ndarray(N, M)): Temp matrix for intermediate storage; cholesky factorization of V.
        X (ndarray(N, M)): Returned matrix.
        V (ndarray(N, N)): Variance matrix.
        B (ndarray(N, M)): Second matrix.
        
    Returns:
        (tuple):
        - **U** (ndarray(N, M)): Temp matrix.
        - **X** (ndarray(N, M)): X in :math:`VX = B`.

    """
    # get dimensions
    cdef int n = V.shape[0], nrhs = B.shape[1], lda = n, ldb = n, info
    cdef char * uplo = 'U'
    mat_copy(U, V)  # operates in-place, so this prevents V from being overwritten
    dpotrf(& uplo[0], & n, & U[0, 0], & lda, & info)
    # solve system with cholesky factor
    mat_copy(X, B)  # again prevents overwriting
    dpotrs(& uplo[0], & n, & nrhs, & U[0, 0], & lda, & X[0, 0], & ldb, & info)
    return
