# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

import numpy as np
import numpy.testing as npt

from spectre.DataStructures.Tensor import DataVector, tnsr
from spectre.PointwiseFunctions.AnalyticSolutions.GeneralRelativity import (
    KerrSchild,
)


class TestGrSolutions(unittest.TestCase):
    def test_kerr_schild(self):
        solution = KerrSchild(mass=1.0, dimensionless_spin=[0.0, 0.0, 0.0])
        # Check some quantities at the horizon
        (lapse, trace_K) = solution.variables(
            np.array([[2.0], [0.0], [0.0]]),
            ["Lapse", "TraceExtrinsicCurvature"],
        ).values()
        npt.assert_allclose(lapse, np.sqrt(0.5))
        npt.assert_allclose(trace_K, 10.0 * (1.0 / 8.0) ** (3.0 / 2.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
