# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.DataStructures import DataVector
import unittest


class TestDataVector(unittest.TestCase):
    def test_math_double(self):
        a = DataVector(5, 6.7)
        self.assertEqual(a + 2.3, DataVector(5, 9.0))
        self.assertEqual(2.3 + a, DataVector(5, 9.0))
        self.assertEqual(a - 2.3, DataVector(5, 4.4))
        self.assertEqual(2.3 - a, DataVector(5, -4.4))
        self.assertEqual(2.0 * a, DataVector(5, 2.0 * 6.7))
        self.assertEqual(a * 2.0, DataVector(5, 2.0 * 6.7))
        self.assertEqual(a / 2.0, DataVector(5, 6.7 / 2.0))
        self.assertEqual(2.0 / a, DataVector(5, 2.0 / 6.7))

        b = DataVector(2)
        b[0] = 7.2
        b[1] = -3.9
        self.assertEqual((b + 2.3)[0], 9.5)
        self.assertEqual((b + 2.3)[1], -1.6)
        self.assertEqual((2.3 + b)[0], 9.5)
        self.assertEqual((2.3 + b)[1], -1.6)

        self.assertEqual((b - 2.3)[0], 4.9)
        self.assertAlmostEqual((b - 2.3)[1], -6.2)
        self.assertEqual((2.3 - b)[0], -4.9)
        self.assertAlmostEqual((2.3 - b)[1], 6.2)

        self.assertEqual((2.0 * b)[0], 14.4)
        self.assertEqual((2.0 * b)[1], -7.8)
        self.assertEqual((b * 2.0)[0], 14.4)
        self.assertEqual((b * 2.0)[1], -7.8)

        self.assertEqual((2.0 / b)[0], 0.2777777777777778)
        self.assertEqual((2.0 / b)[1], -0.5128205128205129)
        self.assertEqual((b / 2.0)[0], 3.6)
        self.assertEqual((b / 2.0)[1], -1.95)

    def test_math_datavector(self):
        a = DataVector(5, 6.7)
        b = DataVector(5, 8.7)
        self.assertEqual(a + b, DataVector(5, 6.7 + 8.7))
        self.assertEqual(a - b, DataVector(5, 6.7 - 8.7))
        self.assertEqual(a * b, DataVector(5, 6.7 * 8.7))
        self.assertEqual(a / b, DataVector(5, 6.7 / 8.7))

        c = DataVector(2)
        c[0] = 1.5
        c[1] = 2.5
        d = DataVector(2)
        d[0] = 3.0
        d[1] = 7.5
        self.assertEqual((c + d)[0], 4.5)
        self.assertEqual((c + d)[1], 10.0)
        self.assertEqual((c - d)[0], -1.5)
        self.assertEqual((c - d)[1], -5.0)
        self.assertEqual((c * d)[0], 4.5)
        self.assertEqual((c * d)[1], 18.75)
        self.assertEqual((c / d)[0], 0.5)
        self.assertEqual((c / d)[1], 1.0 / 3.0)

    def test_negate(self):
        a = DataVector(5, 6.7)
        self.assertEqual(-a, DataVector(5, -6.7))
        b = DataVector(2)
        b[0] = 1.5
        b[1] = 3.7
        self.assertEqual((-b)[0], -1.5)
        self.assertEqual((-b)[1], -3.7)

    def test_bounds_check(self):
        a = DataVector(5, 6.7)
        self.assertRaises(RuntimeError, lambda: a[5])

        def assignment_test():
            a[5] = 8

        self.assertRaises(RuntimeError, assignment_test)

    def test_output(self):
        a = DataVector(5, 6.7)
        self.assertEqual(str(a), "(6.7,6.7,6.7,6.7,6.7)")
        b = DataVector(2)
        b[0] = 3.8
        b[1] = 7.2
        self.assertEqual(str(b), "(3.8,7.2)")


if __name__ == '__main__':
    unittest.main(verbosity=2)
