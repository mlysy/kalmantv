import numpy as np
cimport numpy as np
cimport cython


cpdef _omp_init():
    """
    Seems to be an issue with `-fopenmp` on macOS llvm clang which looks like this:
    ```
ImportError: dlopen(/mypylib/site-packages/kalmantv/eigen/kalmantv.cpython-38-darwin.so, 2): Symbol not found: ___kmpc_fork_call
  Referenced from: /mypylib/site-packages/kalmantv/eigen/kalmantv.cpython-38-darwin.so
  Expected in: flat namespace
 in /mypylib/site-packages/kalmantv/eigen/kalmantv.cpython-38-darwin.so
    ```
    Goes away if other cython code is loaded first, so that's what this function is for.
    """
    cdef int x = 0
    return x
