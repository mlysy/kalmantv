import numpy as np
cimport numpy as np
cimport cython
cimport scipy.linalg.cython_blas as blas
from scipy.linalg.cython_lapack cimport dpotrf, dpotrs, dlacpy

DTYPE = np.double
ctypedef np.double_t DTYPE_t

cpdef vec_copy(const double[::1] x,
               double[::1] y):
    """
    Copies vector x to y.

    Args:
        x (ndarray(N)): Vector x.
        y (ndarray(N)): Returned vector.
    
    Returns:
        (ndarray(M)): Copied vector y.
    
    """
    cdef int N = len(x), incx = 1, incy = 1
    blas.dcopy(&N, &x[0], &incx, &y[0], &incy)
    return

cpdef mat_copy(const double[::1, :] A,
               double[::1, :] B):
    """
    Copies Matrix A to B.

    Args:
        A (ndarray(N)): Matrix A.
        B (ndarray(N)): Returned matrix.
    
    Returns:
        (ndarray(M)): Copied matrix B.
    
    """
    cdef char* uplo = 'A'
    cdef int M = A.shape[0], N = A.shape[1], lda = M, ldb=M
    dlacpy(&uplo[0], &M, &N, &A[0, 0], &lda, &B[0, 0], &ldb)
    return

cpdef vec_add(const double alpha, 
              const double[::1] x,
              double[::1] y):
    """
    Calculates :math:`y = \\alpha x + y`.

    Args:
        alpha (double): Scalar :math:`\\alpha`.
        x (ndarray(N)): Vector x.
        y (ndarray(N)): Returned vector.
    
    Returns:
        (ndarray(M)): :math:`y = \\alpha x + y`.
    
    """
    cdef int N = len(x), incx = 1, incy = 1
    blas.daxpy(&N, &alpha, &x[0], &incx, &y[0], &incy)
    return

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef mat_add(const double alpha,
              const double[::1, :] A,
              const double beta,
              double[::1, :] B):
    r"""
    Calculates :math:`B = \alpha A + \beta B`.
    """
    cdef int M = A.shape[0], N = A.shape[1]
    for i in range(M):
        for j in range(N):
            B[i, j] = alpha*A[i, j] + beta*B[i, j]
    return

cpdef mat_vec_mult(char* trans, 
                   const double alpha, 
                   const double[::1, :] A, 
                   const double[::1] x, 
                   const double beta,
                   double[::1] y):
    """
    Calculates :math:`y = \\alpha A x + \\beta y`.

    Args:
        trans (char): Specifies if matrix A should be transposed.
        alpha (double): Scalar :math:`\\alpha`.
        A (ndarray(M, N)): Matrix A.
        x (ndarray(N)): Vector x.
        beta (double): Scalar :math:`\\beta`.
        y (ndarray(M)): Returned vector.

    Returns:
        (ndarray(M)): :math:`y = \\alpha A x + \\beta y`.

    """
    cdef int M = A.shape[0], N = A.shape[1], lda = M, incx = 1, incy = 1
    blas.dgemv(&trans[0], &M, &N, &alpha, &A[0, 0], &lda, &x[0], &incx, &beta, &y[0], &incy)
    return

cpdef tri_vec_mult(char* uplo,
                   char* trans,
                   char* diag, 
                   const double[::1, :] A, 
                   const double[::1] x):
    """
    Calculates :math:`x = A x` where A is a triangular matrix.

    Args:
        uplo (char): Specifies if matrix A is upper or lower triangular.
        trans (char): Specifies if matrix A should be transposed.
        diag (char): Specifies if matrix A is unit triangular.
        A (ndarray(M, N)): Matrix A.
        x (ndarray(N)): Vector x.

    Returns:
        (ndarray(M)): :math:`x = A x`.

    """
    cdef int N = A.shape[0], lda = N, incx = 1
    blas.dtrmv(&uplo[0], &trans[0], &diag[0], &N, &A[0, 0], &lda, &x[0], &incx)
    return

cpdef mat_mult(char* transa, 
               char* transb, 
               const double alpha, 
               const double[::1, :] A, 
               const double[::1, :] B,
               const double beta, 
               double[::1, :] C):
    """
    Calculates :math:`C = \\alpha A B + \\beta C`.

    Args:
        transa (char): Specifies if matrix A should be transposed.
        transb (char): Specifies if matrix B should be transposed.
        alpha (double): Scalar :math:`\\alpha`.
        A (ndarray(M, K)): First matrix.
        B (ndarray(K, N)): Second matrix.
        beta (double): Scalar :math:`\\beta`.
        C (ndarray(M, N)): Returned matrix.

    Returns:
        (ndarray(M, N)): :math:`C = \\alpha A B + \\beta C`.

    """
    # get dimensions
    cdef int M, N, K, lda, ldb, ldc
    if transa==b'N':
        M = A.shape[0]
        K = A.shape[1]
        lda = M
    else:
        M = A.shape[1]
        K = A.shape[0]
        lda = K
    
    if transb==b'N':
        N = B.shape[1]
        ldb = K
    else:
        N = B.shape[0]
        ldb = N
    ldc = M
    blas.dgemm(&transa[0], &transb[0], &M, &N, &K, &alpha, &A[0, 0], &lda, &B[0, 0], &ldb, &beta, &C[0, 0], &ldc)
    return

cpdef mat_triple_mult(char* transa, 
                      char* transb, 
                      char* transc, 
                      const double alpha, 
                      const double[::1, :] A, 
                      const double[::1, :] B, 
                      double[::1, :] temp, 
                      const double[::1, :] C, 
                      const double beta,
                      double[::1, :] D):
    """
    Calculates :math:`D = \\alpha A B C + \\beta D`.

    Args:
        transa (char): Specifies if matrix A should be transposed.
        transb (char): Specifies if matrix B should be transposed.
        transc (char): Specifies if matrix C should be transposed.
        alpha (double): Scalar :math:`\\alpha`.
        A (ndarray(M, K)): First matrix.
        B (ndarray(K, L)): Second matrix.
        temp (ndarray(M, L)): Temp matrix for intermediate matrix multiplication; :math:`AB`.
        C (ndarray(L, N)): Third matrix.
        beta (double): Scalar :math:`\\beta`.
        D (ndarray(L, N)): Returned matrix.

    Returns:
        (ndarray(M, N)): :math:`D = \\alpha A B C + \\beta D`.
    
    """
    # Temp alpha, beta
    cdef int alpha1 = 1, beta1 = 0
    # Temp trans
    cdef char* trans1 = 'N'
    mat_mult(transa, transb, alpha1, A, B, beta1, temp)
    mat_mult(trans1, transc, alpha, temp, C, beta, D)
    return

cpdef chol_fact(const double[::1, :] V,
                double[::1, :] U):
    """
    Computes the cholesky factorization of variance matrix V.
    
    Args:
        V (ndarray(N, N)): Variance matrix.
        U (ndarray(N, M)): Returned matrix.
    
    Returns:
        (ndarray(N, M)): Cholesky factorization of variance matrix V

    """
    cdef char* uplo = 'L'
    cdef int N = V.shape[0], lda = N, info
    mat_copy(V, U) # operates in-place, so this prevents V from being overwritten
    dpotrf(&uplo[0], &N, &U[0,0], &lda, &info)
    return 

cpdef solveV(const double[::1, :] V, 
             const double[::1, :] B, 
             double[::1, :] U,
             double[::1, :] X):
    """
    Solves X in :math:`VX = B`, where V is a variance matrix.
    
    Args:
        V (ndarray(N, N)): Variance matrix.
        B (ndarray(N, M)): Second matrix.
        U (ndarray(N, M)): Temp matrix for intermediate storage.
        X (ndarray(N, M)): Returned matrix.
    
    Returns:
        (ndarray(N, M)): X in :math:`VX = B`.

    """
    # get dimensions
    cdef int n=V.shape[0], nrhs=B.shape[1], lda=n, ldb=n, info
    cdef char* uplo='U'
    # cholesky factor
    mat_copy(V, U) # operates in-place, so this prevents V from being overwritten
    dpotrf(&uplo[0], &n, &U[0,0], &lda, &info)
    # solve system with cholesky factor
    mat_copy(B, X)  # again prevents overwriting
    dpotrs(&uplo[0], &n, &nrhs, &U[0,0], &lda, &X[0,0], &ldb, &info)
    return
