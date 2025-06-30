#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import argparse
import logging
import unittest

import numpy as np
import numpy.testing as npt
from AnalyticTestVolumeUtils import CheckCceVolumeBase, load_cce_volume_field


class CheckPsi2Case(CheckCceVolumeBase):
    """Unit tests for the analytically expected value of the Newman-Penrose
    Weyl scalar Psi_2 in the Schwarzschild background."""

    def setUp(self):
        super().setUp()
        # We can do extra setup here... but don't need to

    def test_psi2(self):
        psi2 = load_cce_volume_field(self.vol_f, self.vol_obsids[0], "Psi2")

        one_over_r_cubed_vol = np.zeros_like(psi2)
        one_over_r_cubed_vol[:, 0] = np.sqrt(4.0 * np.pi) * self.one_over_r**3

        try:
            npt.assert_allclose(
                psi2,
                -one_over_r_cubed_vol,
                atol=self.atol,
                rtol=self.rtol,
            )
        except AssertionError as e:
            np.set_printoptions(precision=16)
            print(f"DESIRED: { -one_over_r_cubed_vol }")
            print(f"ACTUAL: {psi2}")
            raise AssertionError(
                "Test data is not equal to the expected data"
            ) from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-filename")
    parser.add_argument("--run-directory")
    parser.add_argument("--cmake-source-directory")
    parser.add_argument("--cmake-bin-directory")
    parser.add_argument("--atol", type=float, default=1.0e-8)
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    logging.basicConfig(level=logging.INFO)
    duplicate_test_case, remaining_args = parser.parse_known_args(
        namespace=CheckPsi2Case
    )
    del duplicate_test_case
    # Use of full command-line arguments breaks the unit-test framework
    # (which needs to take its own command-line arguments), so we only pass
    # on the remaining args after we've retrieved used ones
    unittest.main(argv=[parser.prog] + remaining_args, verbosity=2)
