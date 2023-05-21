# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

import numpy as np


def jacobian_diagnostic(analytic_jacobian, numeric_jacobian_transpose):
    analytic_sum = np.abs(analytic_jacobian).sum(axis=0)
    numeric_sum = np.abs(numeric_jacobian_transpose).sum(axis=1)
    jac_diag = 1.0 - analytic_sum / numeric_sum
    return jac_diag


if __name__ == "__main__":
    unittest.main(verbosity=2)
