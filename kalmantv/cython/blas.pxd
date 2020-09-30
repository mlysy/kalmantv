cpdef void vec_copy(double[::1] y,
                    const double[::1] x)
cpdef void mat_copy(double[::1, :] B,
                    const double[::1, :] A)
cpdef void vec_add(double[::1] y,
                   const double alpha, 
                   const double[::1] x)
cpdef void mat_add(double[::1, :] B,
                   const double alpha,
                   const double beta,
                   const double[::1, :] A)
cpdef void mat_vec_mult(double[::1] y,
                        char* trans,
                        const double alpha, 
                        const double beta,
                        const double[::1, :] A, 
                        const double[::1] x)
cpdef void tri_vec_mult(const double[::1] x, 
                        char* uplo,
                        char* trans,
                        char* diag, 
                        const double[::1, :] A)
cpdef void mat_mult(double[::1, :] C,
                    char* transa, 
                    char* transb, 
                    const double alpha, 
                    const double beta,
                    const double[::1, :] A, 
                    const double[::1, :] B)
cpdef void mat_triple_mult(double[::1, :] D,
                           double[::1, :] temp, 
                           char* transa, 
                           char* transb, 
                           char* transc, 
                           const double alpha, 
                           const double beta,
                           const double[::1, :] A, 
                           const double[::1, :] B, 
                           const double[::1, :] C)
cpdef void chol_fact(double[::1, :] U,
                     const double[::1, :] V)
cpdef void solveV(double[::1, :] U,
                  double[::1, :] X,
                  const double[::1, :] V, 
                  const double[::1, :] B)
                  