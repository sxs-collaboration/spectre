# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def test_none():
    return None


def test_numeric(a, b):
    return a * b


def test_vector(a, b):
    test = list()
    test.append(a[0] * b[0])
    test.append(a[1] * b[1])
    return test


def test_unordered_map(a, b):
    test = {}
    test["0"] = a["0"] * b["0"]
    test["1"] = a["1"] * b["1"]
    return test


def test_tuple(a, b):
    if a[0] != 0.35 or a[1] != "test 1" or b[0] != "test 2" or b[1] != 100:
        raise RuntimeError('Failed test_tuple')
    return 0.48, "test 3"


def test_tuple_list(a, b):
    if a[0] != 0.35 or a[1] != "test 1" or b[0] != "test 2" or b[1] != 100:
        raise RuntimeError('Failed test_tuple')
    return 0.48, [0.98, "test 4"]


def test_string(a):
    if a != "test string":
        raise RuntimeError('Failed string test')
    return "back test string"


# These are used to check that converting to a DataVector throws


def two_dim_ndarray():
    import numpy as np
    return np.array([[1., 2.], [3., 4]])


def ndarray_of_floats():
    import numpy as np
    return np.array([1., 2.], dtype='float32')


# These are used to check converting to a Tensor works


def scalar_from_double():
    return 0.8


def scalar_from_ndarray():
    return np.array(0.8)


def vector():
    return np.array([3., 4., 5., 6.])


def tnsr_ia():
    return np.array([[i + 2 * j + 1. for j in range(4)] for i in range(3)])


def tnsr_AA():
    return np.array([[i + j + 1. for j in range(4)] for i in range(4)])


def tnsr_iaa():
    return np.array([[[2. * (k + 1) * (j + 1) + i + 1. for k in range(4)]
                      for j in range(4)] for i in range(3)])


def tnsr_aia():
    a = np.array([[[2. * (k + 1) * (i + 1) + j + 1.5 for k in range(4)]
                   for j in range(3)] for i in range(4)])
    return a


def tnsr_aBcc():
    return np.array([[[[3. * i + j + (k + 1) * (l + 1) + 1. for l in range(4)]
                       for k in range(4)] for j in range(4)]
                     for i in range(4)])


# Test conversion from Tensor to numpy array works


def convert_scalar_to_ndarray_successful(a):
    return bool(np.all(a == scalar_from_ndarray()))


def convert_scalar_to_double_unsuccessful(a):
    return not isinstance(scalar_from_double(), type(a))


def convert_vector_successful(a):
    return bool(np.all(a == vector()))


def convert_tnsr_ia_successful(a):
    return bool(np.all(a == tnsr_ia()))


def convert_tnsr_AA_successful(a):
    return bool(np.all(a == tnsr_AA()))


def convert_tnsr_iaa_successful(a):
    return bool(np.all(a == tnsr_iaa()))


def convert_tnsr_aia_successful(a):
    return bool(np.all(a == tnsr_aia()))


def convert_tnsr_aBcc_successful(a):
    return bool(np.all(a == tnsr_aBcc()))


def test_function_of_time(x, t):
    return 2 * x[0] + x[1] - x[2] - t


# Used to test tensor of datavectors
def identity(a):
    return a


def test_einsum(scalar, t_A, t_ia, t_AA, t_iaa):
    return scalar * np.einsum("a,ia->i", t_A, t_ia) + np.einsum(
        "ab, iab->i", t_AA, t_iaa)


def check_single_not_null0(t0):
    return t0 + 5.0


def check_single_not_null1(t0, t1):
    return t0 + t1


def check_single_not_null2(t0, t1):
    return np.sqrt(t0) + 1.0 / np.sqrt(-t1)


def check_double_not_null0_result0(t0):
    return t0 + 5.0


def check_double_not_null0_result1(t0):
    return 2.0 * t0 + 5.0


def check_double_not_null1_result0(t0, t1):
    return t0 + t1


def check_double_not_null1_result1(t0, t1):
    return 2.0 * t0 + t1


# [python_two_not_null]
def check_double_not_null2_result0(t0, t1):
    return np.sqrt(t0) + 1.0 / np.sqrt(-t1)


def check_double_not_null2_result1(t0, t1):
    return 2.0 * t0 + t1
    # [python_two_not_null]


def check_by_value0(t0):
    return t0 + 5.0


def check_by_value1(t0, t1):
    return t0 + t1


def check_by_value2(t0, t1):
    return np.sqrt(t0) + 1.0 / np.sqrt(-t1)


# the below are used both by value and by not_null checks
def check_by_value1_class(t0, a):
    return t0 + 5.0 * a


def check_by_value2_class(t0, a, b):
    return t0 + 5.0 * a + b


def check_by_value3_class(t0, a, b, c):
    return t0 + 5.0 * a + b + c[0] - 2.0 * c[1] - c[2]


def check2_by_value1_class(t0, t1, a):
    return t0 + t1 + 5.0 * a


def check2_by_value2_class(t0, t1, a, b):
    return t0 + 5.0 * a + t1 * b


def check2_by_value3_class(t0, t1, a, b, c):
    return t0 * c[0] + 5.0 * a + t1 * b + c[1] - c[2]


def check2_by_value1_class1(t0, t1, a):
    return 2.0 * t0 + t1 + 5.0 * a


def check2_by_value2_class1(t0, t1, a, b):
    return 2.0 * t0 + 5.0 * a + t1 * b


def permute_array(a):
    return [a[2], a[0], a[1]]


def check_solution_scalar(x, t, a, b):
    return np.dot(x, b) + a - t


def check_solution_vector(x, t, a, b):
    # b is passed in as a list so must be explicitly converted to an array
    return a * x - t * np.array(b)
