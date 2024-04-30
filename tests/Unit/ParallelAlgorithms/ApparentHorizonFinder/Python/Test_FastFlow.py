# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

from spectre.ApparentHorizonFinder import FastFlow, FlowType


class TestFastFlow(unittest.TestCase):
    def test_fast_flow(self):
        fast_flow = FastFlow(
            FlowType.Fast,
            alpha=1.0,
            beta=0.5,
            abs_tol=1e-12,
            truncation_tol=0.01,
            divergence_tol=1.2,
            divergence_iter=5,
            max_its=100,
        )
        # Can't test any iterations until we bind a Schwarzschild or Kerr
        # solution in Python, or have numeric data with a horizon stored
        # somewhere.


if __name__ == "__main__":
    unittest.main(verbosity=2)
