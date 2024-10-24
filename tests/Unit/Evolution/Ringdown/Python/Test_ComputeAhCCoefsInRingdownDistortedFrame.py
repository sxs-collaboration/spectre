# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import shutil
import unittest
from pathlib import Path

import spectre.IO.H5 as spectre_h5
from spectre import Spectral
from spectre.DataStructures import DataVector, ModalVector
from spectre.Evolution.Ringdown.ComputeAhCCoefsInRingdownDistortedFrame import (
    compute_ahc_coefs_in_ringdown_distorted_frame,
)
from spectre.Informer import unit_test_build_path
from spectre.SphericalHarmonics import Strahlkorper, ylm_legend_and_data
from spectre.support.Logging import configure_logging


class TestComputeAhCCoefs(unittest.TestCase):
    def test_compute_ahc_coefs_in_ringdown_distorted_frame(self):
        # Building a fake directory to hold fake reduction data
        self.test_dir = Path(
            unit_test_build_path(), "Unit/Evolution/Ringdown/Python/Ringdown"
        )
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.inspiral_reduction_data = self.test_dir / "BbhReductions.h5"
        shape_coefs = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        times = [4990.0, 4992.0, 4994.0, 4996.0, 4998.0, 5000.0]
        time_to_match = 5000.0
        ahc_center = [0.0, 0.0, 0.0]
        ahc_lmax = 2
        with spectre_h5.H5File(
            str(self.inspiral_reduction_data.resolve()), "a"
        ) as reduction_file:
            legend = [
                "Time",
                "InertialExpansionCenter_x",
                "InertialExpansionCenter_y",
                "InertialExpansionCenter_z",
                "Lmax",
                "coef(0,0)",
                "coef(1,-1)",
                "coef(1,0)",
                "coef(1,1)",
                "coef(2,-2)",
                "coef(2,-1)",
                "coef(2,0)",
                "coef(2,1)",
                "coef(2,2)",
            ]
            reduction_dat = reduction_file.try_insert_dat(
                "ObservationAhC_Ylm.dat", legend, 0
            )
            for x in range(0, 5):
                reduction_dat.append(
                    [
                        [
                            times[x],
                            ahc_center[0],
                            ahc_center[1],
                            ahc_center[2],
                            ahc_lmax,
                            shape_coefs[x],
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ]
                    ]
                )
            reduction_file.close_current_object()

        exp_inner_func_with_2_derivs = [1.0, 0.0, 0.0]
        exp_outer_boundary_func_with_2_derivs = [1.0, -1e-6, 0.0]
        rot_func_with_2_derivs = [
            [0.0, 0.0, 0.0, 1.0],
            [0.15, 0.0, 0.0, 0.02],
            [0.06, 0.0, 0.0, 0.03],
        ]
        ringdown_ylm_coefs, ringdown_ylm_legend = (
            compute_ahc_coefs_in_ringdown_distorted_frame(
                ahc_reductions_path=str(self.inspiral_reduction_data),
                ahc_subfile="ObservationAhC_Ylm.dat",
                exp_func_and_2_derivs=exp_inner_func_with_2_derivs,
                exp_outer_bdry_func_and_2_derivs=(
                    exp_outer_boundary_func_with_2_derivs
                ),
                rot_func_and_2_derivs=rot_func_with_2_derivs,
                number_of_ahc_finds_for_fit=5,
                match_time=time_to_match,
                settling_timescale=10.0,
                zero_coefs_eps=None,
            )
        )
        # Expected fit should be a line
        expected_fit_ahc_coefs = [
            10,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        expected_fit_dt_ahc_coefs = [
            0.5,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        expected_fit_dt2_ahc_coefs = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        expected_fit_ahc_coefs_mv = ModalVector(expected_fit_ahc_coefs)
        expected_fit_dt_ahc_coefs_mv = ModalVector(expected_fit_dt_ahc_coefs)
        expected_fit_dt2_ahc_coefs_mv = ModalVector(expected_fit_dt2_ahc_coefs)
        expected_ahc_strahlkorper = Strahlkorper(
            ahc_lmax, ahc_lmax, expected_fit_ahc_coefs_mv, ahc_center
        )
        expected_dt_ahc_strahlkorper = Strahlkorper(
            ahc_lmax, ahc_lmax, expected_fit_dt_ahc_coefs_mv, ahc_center
        )
        expected_dt2_ahc_strahlkorper = Strahlkorper(
            ahc_lmax, ahc_lmax, expected_fit_dt2_ahc_coefs_mv, ahc_center
        )
        # These are bad legends because they say InertialExpansionCenter instead
        # of Distorted.
        bad_legend_ahc, expected_ahc_ylm_coefs = ylm_legend_and_data(
            expected_ahc_strahlkorper, time_to_match, ahc_lmax
        )
        bad_legend_dt_ahc, expected_dt_ahc_ylm_coefs = ylm_legend_and_data(
            expected_dt_ahc_strahlkorper, time_to_match, ahc_lmax
        )
        bad_legend_dt2_ahc, expected_dt2_ahc_ylm_coefs = ylm_legend_and_data(
            expected_dt2_ahc_strahlkorper, time_to_match, ahc_lmax
        )
        expected_legends_ahc = [
            "Time",
            "DistortedExpansionCenter_x",
            "DistortedExpansionCenter_y",
            "DistortedExpansionCenter_z",
            "Lmax",
            "coef(0,0)",
            "coef(1,-1)",
            "coef(1,0)",
            "coef(1,1)",
            "coef(2,-2)",
            "coef(2,-1)",
            "coef(2,0)",
            "coef(2,1)",
            "coef(2,2)",
        ]
        for x in range(0, len(expected_ahc_ylm_coefs)):
            self.assertAlmostEqual(
                first=expected_ahc_ylm_coefs[x],
                second=ringdown_ylm_coefs[0][x],
                places=11,
            )
            self.assertAlmostEqual(
                first=expected_dt_ahc_ylm_coefs[x],
                second=ringdown_ylm_coefs[1][x],
                places=11,
            )
            self.assertAlmostEqual(
                first=expected_dt2_ahc_ylm_coefs[x],
                second=ringdown_ylm_coefs[2][x],
                places=11,
            )
        self.assertNotEqual(bad_legend_ahc, ringdown_ylm_legend[0])
        self.assertNotEqual(bad_legend_dt_ahc, ringdown_ylm_legend[1])
        self.assertNotEqual(bad_legend_dt2_ahc, ringdown_ylm_legend[2])
        self.assertEqual(expected_legends_ahc, ringdown_ylm_legend[0])
        self.assertEqual(expected_legends_ahc, ringdown_ylm_legend[1])
        self.assertEqual(expected_legends_ahc, ringdown_ylm_legend[2])


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
