cpdef void vec_copy(const double[::1] x,
                    double[::1] y)
cpdef void mat_copy(const double[::1, :] A,
                    double[::1, :] B)
cpdef void vec_add(const double alpha, 
                   const double[::1] x,
                   double[::1] y)
cpdef void mat_add(const double alpha,
                   const double[::1, :] A,
                   const double beta,
                   double[::1, :] B)
cpdef void mat_vec_mult(char* trans, 
                        const double alpha, 
                        const double[::1, :] A, 
                        const double[::1] x, 
                        const double beta,
                        double[::1] y)
cpdef void tri_vec_mult(char* uplo,
                        char* trans,
                        char* diag, 
                        const double[::1, :] A, 
                        const double[::1] x)
cpdef void mat_mult(char* transa, 
                    char* transb, 
                    const double alpha, 
                    const double[::1, :] A, 
                    const double[::1, :] B,
                    const double beta, 
                    double[::1, :] C)
cpdef void mat_triple_mult(char* transa, 
                           char* transb, 
                           char* transc, 
                           const double alpha, 
                           const double[::1, :] A, 
                           const double[::1, :] B, 
                           double[::1, :] temp, 
                           const double[::1, :] C, 
                           const double beta,
                           double[::1, :] D)
cpdef void chol_fact(const double[::1, :] V,
                     double[::1, :] U)
cpdef void solveV(const double[::1, :] V, 
                  const double[::1, :] B, 
                  double[::1, :] U,
                  double[::1, :] X)
                  