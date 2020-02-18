cimport cython
import numpy as np
cimport numpy as np
cimport scipy.linalg.cython_blas as blas

DTYPE = np.double
ctypedef np.double_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef mat_vec_mult(double[::1, :] A, double[::1] x, double[::1] y):
    """
    Calculates y=A*x.

    Args:
        A (ndarray(M, N)): Matrix.
        x (ndarray(N)): Input vector.
        y (ndarray(M)): Returned vector.
        
    Returns:
        (ndarray(M)): y=A*x.

    """
    cdef int M = A.shape[0] # Rows
    cdef int N = A.shape[1] # Columns
    cdef int lda = M
    cdef int incx = 1 #increments of x
    cdef int incy = 1 #increments of y
    cdef double alpha = 1.0
    cdef double beta = 0.0
    blas.dgemv("N", &M, &N, &alpha, &A[0,0], &lda, &x[0], &incx, &beta, &y[0], &incy)
    return

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef mat_mult(double[::1, :] A, double[::1, :] B, double[::1, :] C):
    """
    Calculates C=A*B.

    Args:
        A (ndarray(M, K)): First matrix.
        B (ndarray(K, N)): Second matrix.
        C (ndarray(M, N)): Returned matrix.
    Returns:
        (ndarray(M, N)): C=A*B.

    """
    cdef int M = A.shape[0] # Rows of A
    cdef int N = B.shape[1] # Columns of B
    cdef int K = A.shape[1] # Columns of A
    cdef double alpha = 1.0
    cdef double beta = 0.0
    cdef int lda = M
    cdef int ldb = K
    cdef int ldc = M
    #cdef np.ndarray[DTYPE_t, ndim=2] C = np.zeros((M,N), dtype=DTYPE, order='F')
    blas.dgemm("N", "N", &M, &N, &K, &alpha, &A[0,0], &lda, &B[0,0], &ldb, &beta, &C[0,0], &ldc)
    return
    