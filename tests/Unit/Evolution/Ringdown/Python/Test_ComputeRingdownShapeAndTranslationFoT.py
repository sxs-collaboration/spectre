# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import math
import shutil
import unittest
from pathlib import Path

import numpy as np

import spectre.IO.H5 as spectre_h5
from spectre import Spectral
from spectre.DataStructures import DataVector, ModalVector
from spectre.Domain import (
    PiecewisePolynomial2,
    PiecewisePolynomial3,
    QuaternionFunctionOfTime,
    serialize_domain,
    serialize_functions_of_time,
)
from spectre.Domain.Creators import BinaryCompactObject, DomainCreator3D
from spectre.Domain.Creators.TimeDependentOptions import (
    BinaryCompactObjectTimeDependentOptions,
    ExpansionMapOptions,
    RotationMapOptions,
    TranslationMapOptions,
)
from spectre.Evolution.Ringdown.ComputeRingdownShapeAndTranslationFoT import (
    compute_ringdown_shape_and_translation_fot,
)
from spectre.Informer import unit_test_build_path
from spectre.IO.H5 import ElementVolumeData, TensorComponent
from spectre.SphericalHarmonics import Frame, Strahlkorper, ylm_legend_and_data
from spectre.support.Logging import configure_logging


class TestComputeAhCCoefs(unittest.TestCase):
    def test_compute_ringdown_shape_and_translation_fot(self):
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

        fot_dict = {}
        fot_dict["Expansion"] = [1.0, 0.0, 0.0]
        fot_dict["ExpansionOuterBoundary"] = [1.0, -1e-6, 0.0]
        fot_dict["Rotation"] = [
            [0.0, 0.0, 0.0, 1.0],
            [0.15, 0.0, 0.0, 0.02],
            [0.06, 0.0, 0.0, 0.03],
        ]
        fot_dict["Translation"] = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]

        # Making volume data for functions of time to be extracted
        rotation_fot = QuaternionFunctionOfTime(
            time=times[0],
            initial_quat_func=[DataVector(size=4, fill=1.0)],
            initial_angle_func=4 * [DataVector(size=3, fill=0.0)],
            expiration_time=math.inf,
        )
        expansion_fot = PiecewisePolynomial3(
            times[0], 4 * [DataVector(size=1, fill=1.0)], math.inf
        )
        expansion_outer_fot = PiecewisePolynomial3(
            times[0], 4 * [DataVector(size=1, fill=1.0)], math.inf
        )
        translation_fot = PiecewisePolynomial2(
            times[0],
            [
                DataVector([1.0, -1.0, 0.5]),
                DataVector([0.2, 0.1, 0.0]),
                DataVector([0.0, 0.0, 0.0]),
            ],
            math.inf,
        )
        serialized_fots = serialize_functions_of_time(
            {
                "Expansion": expansion_fot,
                "ExpansionOuterBoundary": expansion_outer_fot,
                "Rotation": rotation_fot,
                "Translation": translation_fot,
            }
        )

        expansion_map = ExpansionMapOptions([1.0, 1e-4, 0.0], 100.0, 1e-6)
        rotation_map = RotationMapOptions([[0.0, 0.0, 0.0, 1.0]], 100.0)
        translation_map = TranslationMapOptions(
            [[1.0, -1.0, 0.5], [0.2, 0.1, 0.0], [0.0, 0.0, 0.0]]
        )
        bco_time_dependent_options = BinaryCompactObjectTimeDependentOptions(
            times[0],
            expansion_map,
            rotation_map,
            translation_map,
            None,
            None,
            None,
            None,
        )

        binary_domain = BinaryCompactObject(
            inner_radius_a=0.5,
            outer_radius_a=2.0,
            x_coord_a=5.0,
            excise_a=True,
            use_logarithmic_map_a=True,
            inner_radius_b=0.5,
            outer_radius_b=2.0,
            x_coord_b=-5.0,
            excise_b=True,
            use_logarithmic_map_b=True,
            center_of_mass_offset=[0.1, 0.2],
            envelope_radius=50.0,
            outer_radius=600.0,
            cube_scale=1.2,
            initial_refinement=1,
            initial_number_of_grid_points=5,
            use_equiangular_map=True,
            radial_partitioning_outer_shell=[],
            opening_angle_in_degrees=120.0,
            time_dependent_options=bco_time_dependent_options,
        ).create_domain()

        serialized_binary_domain = serialize_domain(binary_domain)
        self.inspiral_volume_data = self.test_dir / "BbhVolume0.h5"
        with spectre_h5.H5File(self.inspiral_volume_data, "w") as volume_file:
            volfile = volume_file.insert_vol("ForContinuation", version=0)
            for x in range(0, 6):
                volfile.write_volume_data(
                    observation_id=x,
                    observation_value=times[x],
                    elements=[
                        ElementVolumeData(
                            element_name="WhatTheFreak",
                            components=[
                                TensorComponent(
                                    "IsGoingOnHere",
                                    np.random.rand(3),
                                ),
                            ],
                            extents=[3],
                            basis=[Spectral.Basis.Legendre],
                            quadrature=[Spectral.Quadrature.GaussLobatto],
                        )
                    ],
                    serialized_domain=serialized_binary_domain,
                    serialized_observation_functions_of_time=serialized_fots,
                )
        volume_file.close_current_object()

        ringdown_ylm_coefs, ringdown_ylm_legend, ahc_translation_fot = (
            compute_ringdown_shape_and_translation_fot(
                path_to_volume_data=str(self.inspiral_volume_data),
                volume_subfile_name="ForContinuation",
                ahc_reductions_path=str(self.inspiral_reduction_data),
                ahc_subfile="ObservationAhC_Ylm.dat",
                evaluated_fot_dict=fot_dict,
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
        expected_ahc_strahlkorper = Strahlkorper[Frame.Inertial](
            ahc_lmax, ahc_lmax, expected_fit_ahc_coefs_mv, ahc_center
        )
        expected_dt_ahc_strahlkorper = Strahlkorper[Frame.Inertial](
            ahc_lmax, ahc_lmax, expected_fit_dt_ahc_coefs_mv, ahc_center
        )
        expected_dt2_ahc_strahlkorper = Strahlkorper[Frame.Inertial](
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
