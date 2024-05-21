# Distributed under the MIT License.
# See LICENSE.txt for details.

import math
import unittest

import numpy as np
import numpy.testing as npt

from spectre.DataStructures import ModalVector


class TestModalVector(unittest.TestCase):
    def test_len(self):
        a = ModalVector(5, 6.7)
        self.assertEqual(len(a), 5)

    def test_math_double(self):
        b = ModalVector([7.2, -3.9])

        self.assertEqual((2.0 * b)[0], 14.4)
        self.assertEqual((2.0 * b)[1], -7.8)
        self.assertEqual((b * 2.0)[0], 14.4)
        self.assertEqual((b * 2.0)[1], -7.8)

        self.assertEqual((b / 2.0)[0], 3.6)
        self.assertEqual((b / 2.0)[1], -1.95)

    def test_math_modalvector(self):
        a = ModalVector(5, 6.7)
        b = ModalVector(5, 8.7)
        self.assertEqual(a + b, ModalVector(5, 6.7 + 8.7))
        self.assertEqual(a - b, ModalVector(5, 6.7 - 8.7))

        c = ModalVector([1.5, 2.5])
        d = ModalVector([3.0, 7.5])
        self.assertEqual((c + d)[0], 4.5)
        self.assertEqual((c + d)[1], 10.0)
        self.assertEqual((c - d)[0], -1.5)
        self.assertEqual((c - d)[1], -5.0)

    def test_negate(self):
        a = ModalVector(5, 6.7)
        self.assertEqual(-a, ModalVector(5, -6.7))
        b = ModalVector([1.5, 3.7])
        self.assertEqual((-b)[0], -1.5)
        self.assertEqual((-b)[1], -3.7)

    def test_bounds_check(self):
        a = ModalVector(5, 6.7)
        self.assertRaises(RuntimeError, lambda: a[5])

        def assignment_test():
            a[5] = 8

        self.assertRaises(RuntimeError, assignment_test)

    def test_output(self):
        a = ModalVector(5, 6.7)
        self.assertEqual(str(a), "(6.7,6.7,6.7,6.7,6.7)")
        b = ModalVector([3.8, 7.2])
        self.assertEqual(str(b), "(3.8,7.2)")

    def test_abs(self):
        a = ModalVector(5, -6.0)
        self.assertEqual(a.abs(), ModalVector(5, 6.0))
        b = ModalVector([1.0, -2.0])
        self.assertEqual(b.abs()[0], 1.0)
        self.assertEqual(b.abs()[1], 2.0)

    def test_numpy_compatibility(self):
        b = np.array([1.0, 2.0, 3.0])
        c = ModalVector([1.0, 2.0, 3.0])
        self.assertTrue(((b + c) == np.array([2.0, 4.0, 6.0])).all())
        x = np.linspace(0, 2 * np.pi, 10)
        npt.assert_allclose(ModalVector(x).abs(), np.abs(x))
        # Convert a ModalVector to a Numpy array
        c_array_copy = np.array(c)
        npt.assert_equal(c_array_copy, b)
        # Changing the copy shouldn't change the ModalVector
        c_array_copy[2] = 4.0
        npt.assert_equal(c, b)
        c_array_reference = np.array(c, copy=False)
        npt.assert_equal(c_array_reference, b)
        # Changing the reference should change the ModalVector as well
        c_array_reference[2] = 4.0
        self.assertEqual(c[2], 4.0)
        # Convert a Numpy array to a ModalVector
        b_dv_copy = ModalVector(b)
        self.assertEqual(b_dv_copy, ModalVector([1.0, 2.0, 3.0]))
        b_dv_copy[2] = 4.0
        self.assertEqual(b[2], 3.0)
        b_dv_reference = ModalVector(b, copy=False)
        b_dv_reference[2] = 4.0
        self.assertEqual(b[2], 4.0)

    def test_iterator(self):
        a = ModalVector([1.0, 2.0, 3.0, 4.0, 5.0])
        for i, val in enumerate(a):
            self.assertEqual(val, a[i])

    def test_size_constructor(self):
        a = ModalVector(3)
        self.assertEqual(len(a), 3)

    def test_list_constructor(self):
        a = ModalVector([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(len(a), 5)
        self.assertEqual(a[3], 4.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
