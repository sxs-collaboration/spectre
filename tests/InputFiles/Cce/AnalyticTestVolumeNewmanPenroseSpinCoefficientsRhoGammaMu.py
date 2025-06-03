#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import argparse
import logging
import os
import re
import unittest

import h5py
import numpy as np
import numpy.testing as npt
import yaml


def load_spin_coeff(vol_f, obsid, name):
    obsid_path = f"/CceVolumeData/VolumeData.vol/ObservationId{obsid}"
    (number_of_radial_shells, l_max, _) = vol_f[f"{obsid_path}/total_extents"][
        ()
    ]
    spin_coeff = vol_f[f"{obsid_path}/{name}"][()]
    spin_coeff = spin_coeff.view(np.complex128).reshape(
        [int(number_of_radial_shells), int((l_max + 1) ** 2)]
    )
    return spin_coeff


class CheckRhoGammaMuTestCase(unittest.TestCase):
    """Unit tests for the analytically expected values of the Newman-Penrose
    spin coefficients rho, gamma, and mu, in the Schwarzschild background."""

    def setUp(self):
        if hasattr(self, "setUp_completed") and self.setUp_completed:
            return

        # Parse the yaml file to know what we're looking for
        with open(self.input_filename, "r") as open_input_file:
            parsed_yaml = list(yaml.safe_load_all(open_input_file))

        # parsed_yaml[0] is the testing part of the stream;
        # parsed_yaml[1] is the input file for the Cce executable

        extraction_R = parsed_yaml[1]["Cce"]["ExtractionRadius"]
        self.vol_f_path = os.path.join(
            self.run_directory,
            parsed_yaml[1]["Observers"]["VolumeFileName"] + ".h5",
        )
        self.vol_f = h5py.File(self.vol_f_path, "r")

        obsid_pat = re.compile(r"ObservationId(?P<obsid>.+)")
        one_minus_y_obsids = [
            obsid_pat.match(k).group("obsid")
            for k in self.vol_f["/CceVolumeData/OneMinusY.vol"].keys()
        ]

        one_minus_y = self.vol_f[
            "/CceVolumeData/OneMinusY.vol/ObservationId"
            + one_minus_y_obsids[0]
            + "/OneMinusY"
        ][()]
        R_over_r = 0.5 * one_minus_y
        self.one_over_r = R_over_r / extraction_R

        self.vol_obsids = [
            obsid_pat.match(k).group("obsid")
            for k in self.vol_f["/CceVolumeData/VolumeData.vol"].keys()
        ]

        self.setUp_completed = True

    def test_rho(self):
        rho = load_spin_coeff(
            self.vol_f, self.vol_obsids[0], "NewmanPenroseRho"
        )

        one_over_r_vol = np.zeros_like(rho)
        one_over_r_vol[:, 0] = np.sqrt(4.0 * np.pi) * self.one_over_r

        try:
            npt.assert_allclose(
                rho,
                -one_over_r_vol / np.sqrt(2.0),
                atol=self.atol,
                rtol=self.rtol,
            )
        except AssertionError as e:
            np.set_printoptions(precision=16)
            print(f"DESIRED: { -one_over_r_vol / np.sqrt(2.0) }")
            print(f"ACTUAL: {rho}")
            raise AssertionError(
                "Test data is not equal to the expected data"
            ) from e

    def test_gamma(self):
        gamma = load_spin_coeff(
            self.vol_f, self.vol_obsids[0], "NewmanPenroseGamma"
        )

        one_over_r_sq_vol = np.zeros_like(gamma)
        one_over_r_sq_vol[:, 0] = np.sqrt(4.0 * np.pi) * self.one_over_r**2

        try:
            npt.assert_allclose(
                gamma,
                one_over_r_sq_vol / np.sqrt(2.0),
                atol=self.atol,
                rtol=self.rtol,
            )
        except AssertionError as e:
            np.set_printoptions(precision=16)
            print(f"DESIRED: { one_over_r_sq_vol / np.sqrt(2.0) }")
            print(f"ACTUAL: {gamma}")
            raise AssertionError(
                "Test data is not equal to the expected data"
            ) from e

    def test_mu(self):
        mu = load_spin_coeff(self.vol_f, self.vol_obsids[0], "NewmanPenroseMu")

        one_over_r_vol = np.zeros_like(mu)
        one_over_r_vol[:, 0] = np.sqrt(4.0 * np.pi) * self.one_over_r
        one_over_r_sq_vol = np.zeros_like(mu)
        one_over_r_sq_vol[:, 0] = np.sqrt(4.0 * np.pi) * self.one_over_r**2

        try:
            npt.assert_allclose(
                mu,
                (2.0 * one_over_r_sq_vol - one_over_r_vol) / np.sqrt(2.0),
                atol=self.atol,
                rtol=self.rtol,
            )
        except AssertionError as e:
            np.set_printoptions(precision=16)
            print(
                "DESIRED:"
                f" {(2.0 * one_over_r_sq_vol - one_over_r_vol) / np.sqrt(2.0)}"
            )
            print(f"ACTUAL: {mu}")
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
        namespace=CheckRhoGammaMuTestCase
    )
    del duplicate_test_case
    # Use of full command-line arguments breaks the unit-test framework
    # (which needs to take its own command-line arguments), so we only pass
    # on the remaining args after we've retrieved used ones
    unittest.main(argv=[parser.prog] + remaining_args, verbosity=2)
