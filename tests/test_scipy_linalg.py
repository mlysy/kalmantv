from numba import njit
from utils import *
import scipy_linalg
import scipy.linalg
import unittest

# Because of numba compilation True takes much longer
_all_tests = False

# n_dim = np.random.randint(5)+1
# V = rand_mat(n_dim)
# b = np.random.randn(n_dim, 3, 10)

# (C, lower) = scipy.linalg.cho_factor(V)
# scipy.linalg.cho_solve((C, lower), b)

# use = njit(scipy.linalg.cho_factor)


# --- simple test functions to check that the overload works -------------------

@njit
def cho_factor_nb(a, lower=False, overwrite_a=False, check_finite=True):
    return scipy.linalg.cho_factor(a, lower, overwrite_a, check_finite)


@njit
def cho_solve_nb(c_and_lower, b, overwrite_b=False, check_finite=True):
    return scipy.linalg.cho_solve(c_and_lower, b, overwrite_b, check_finite)


cho_factor_sp = scipy.linalg.cho_factor
cho_solve_sp = scipy.linalg.cho_solve

# --- test class ---------------------------------------------------------------

# cases = {"order_a": ["F", "C"],
#          "type_a": [np.float32, np.float64],
#          "contiguous_a": [False, True],
#          "overwrite_a": [False, True],
#          "check_finite": [False, True]}

# df = expand_grid(cases)

# print(df)
# print(df.iloc[2])
# print(df.iloc[2]["order_a"])


class TestNumbaSciPy(unittest.TestCase):

    def test_cho_factor(self):
        # test cases
        cases = {"order": ["F", "C"],
                 "dtype": [np.float32, np.float64],
                 # "contiguous_a": [False, True],
                 "lower": [False, True],
                 "overwrite_a": [False, True],
                 "check_finite": [False, True]}
        df_cases = expand_grid(cases)
        # TODO: non-contiguous input check
        n_test = df_cases.shape[0]
        test_set = range(n_test) if _all_tests \
            else np.random.randint(n_test, size=3)
        for i in test_set:
            case = df_cases.iloc[i]
            with self.subTest(case=case):
                a_size = np.random.randint(5)+1
                a1 = rand_mat(a_size,
                              dtype=case["dtype"], order=case["order"])
                a2 = np.copy(a1)
                cf_args = {k: case[k]
                           for k in ("lower", "overwrite_a", "check_finite")}
                (c1, lower1) = cho_factor_nb(a1, **cf_args)
                (c2, lower2) = cho_factor_nb(a2, **cf_args)
                # check values
                self.assertAlmostEqual(rel_err(c1, c2), 0.0)
                # check overwrite
                self.assertEqual(c1 is a1, c2 is a2)
                # check flags
                eq_flags = np.array(
                    [c1.flags[k] == c2.flags[k] for k in ["C", "F", "W", "A"]]
                )
                self.assertTrue(np.all(eq_flags))

    def test_cho_solve(self):
        # test cases
        cases = {"dtype_a": [np.float32, np.float64],
                 # "contiguous_a": [False, True],
                 "lower": [False, True],
                 "dtype_b": [np.float32, np.float64],
                 "order": ["C", "F"],
                 "ndim_b": [1, 2, 3],
                 "overwrite_b": [False, True],
                 "check_finite": [False, True]}
        df_cases = expand_grid(cases)
        n_test = df_cases.shape[0]
        test_set = range(n_test) if _all_tests \
            else np.random.randint(n_test, size=3)
        for i in test_set:
            case = df_cases.iloc[i]
            with self.subTest(case=case):
                a_size = np.random.randint(5)+1
                b_ndim = case["ndim_b"]
                b_shape = np.random.randint(low=1, high=5, size=b_ndim)
                b_shape[0] = a_size
                a = rand_mat(a_size,
                             dtype=case["dtype_a"], order="F")
                b1 = rand_array(b_shape,
                                dtype=case["dtype_b"], order=case["order"])
                b2 = np.copy(b1)
                (c, lower) = cho_factor_sp(a, lower=case["lower"])
                cs_args = {k: case[k]
                           for k in ("overwrite_b", "check_finite")}
                x1 = cho_solve_nb((c, lower), b=b1, **cs_args)
                x2 = cho_solve_nb((c, lower), b=b2, **cs_args)
                # check values
                self.assertAlmostEqual(rel_err(x1, x2), 0.0)
                # check overwrite
                self.assertEqual(x1 is b1, x2 is b2)
                # check flags
                eq_flags = np.array(
                    [x1.flags[k] == x2.flags[k] for k in ["C", "F", "W", "A"]]
                )
                self.assertTrue(np.all(eq_flags))


if __name__ == "__main__":
    unittest.main()

# --- scratch ------------------------------------------------------------------

# n_dim = np.random.randint(5)+1
# b_shape = np.random.randint(low=1, high=5, size=1)
# b_shape[0] = n_dim
# V = rand_mat(n_dim, dtype=np.float64)
# V = np.array(V, dtype=np.float32)
# b = np.asfortranarray(np.random.randn(n_dim, 3, 10), dtype=np.float32)
# b = rand_array(b_shape, dtype=np.float32)
# print("b.shape = \n", b_shape)
# print("b = \n", b)
# breakpoint()

# a = scipy_linalg._asfortranarray(V, overwrite_a=False)
# print("a.flags = \n", a.flags)

# (c1, lower1) = cho_factor_nb(V)
# c2 = scipy_linalg._asfortranarray(V)
# # (c, lower)
# (c2, lower2) = cho_factor_sp(V)

# flag_names = ["C", "F", "W", "A"]

# c1_flags = np.array([c1.flags[name] for name in flag_names])
# c2_flags = np.array([c2.flags[name] for name in flag_names])

# print(np.all(c1_flags == c2_flags))

# print("c1.flags = \n", c1.flags)
# print("scipy_linalg._asfortranarray(V).flags = \n", c2.flags)
# print(c1 - c2)
# print("c2.flags = \n", c2.flags)
# print("np.copy(V).flags \n", np.copy(V).flags)

# cho_factor_nb(V)[0].flags
# cho_factor_sp(V)[0].flags

# (c, lower) = cho_factor_sp(V)
# x1 = cho_solve_nb((c, lower), b)
# x2 = cho_solve_sp((c, lower), b)

# print("c.dtype = \n", c.dtype)
# print("x1.dtype = \n", x1.dtype)
# print("x2.dtype = \n", x2.dtype)
# print("c.flags = \n", c.flags)
# print("x1.flags = \n", x1.flags)
# print("x2.flags = \n", x2.flags)
# print("x1 = \n", x1)
# print("x2 = \n", x2)
# print("rel_err = ", rel_err(x1, x2))
# b1 = cho_solve_sp((np.array(c, dtype=np.float64), lower), b, overwrite_b=True)
