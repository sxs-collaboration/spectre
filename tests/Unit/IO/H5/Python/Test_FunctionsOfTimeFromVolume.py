# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import math
import shutil
import unittest
from pathlib import Path

import numpy as np
import yaml
from click.testing import CliRunner

import spectre.IO.H5 as spectre_h5
from spectre import Spectral
from spectre.DataStructures import DataVector
from spectre.Domain import (
    PiecewisePolynomial3,
    QuaternionFunctionOfTime,
    serialize_functions_of_time,
)
from spectre.Informer import unit_test_build_path
from spectre.IO.H5 import ElementVolumeData, TensorComponent
from spectre.IO.H5.FunctionsOfTimeFromVolume import (
    functions_of_time_from_volume,
)
from spectre.support.Logging import configure_logging


class TestFunctionsOfTimeFromVolume(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(
            unit_test_build_path(), "support/Pipelines/Bbh/Ringdown"
        )
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.bin_dir = Path(unit_test_build_path(), "../../bin").resolve()

        # Making volume data for functions of time to be extracted
        rotation_fot = QuaternionFunctionOfTime(
            0.0,
            [DataVector(size=4, fill=1.0)],
            4 * [DataVector(size=3, fill=0.0)],
            math.inf,
        )
        expansion_fot = PiecewisePolynomial3(
            0.0, 4 * [DataVector(size=1, fill=0.0)], math.inf
        )
        expansion_outer_fot = PiecewisePolynomial3(
            0.0, 4 * [DataVector(size=1, fill=0.0)], math.inf
        )
        translation_fot = PiecewisePolynomial3(
            0.0, 4 * [DataVector(size=3, fill=0.0)], math.inf
        )
        serialized_fots = serialize_functions_of_time(
            {
                "Expansion": expansion_fot,
                "ExpansionOuterBoundary": expansion_outer_fot,
                "Rotation": rotation_fot,
                "Translation": translation_fot,
            }
        )
        self.volume_data = self.test_dir / "BbhVolume0.h5"
        obs_values = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
        with spectre_h5.H5File(self.volume_data, "w") as volume_file:
            volfile = volume_file.insert_vol("ForContinuation", version=0)
            for x in range(0, 5):
                volfile.write_volume_data(
                    observation_id=x,
                    observation_value=obs_values[x],
                    elements=[
                        ElementVolumeData(
                            element_name="arthas",
                            components=[
                                TensorComponent(
                                    "menethil",
                                    np.random.rand(3),
                                ),
                            ],
                            extents=[3],
                            basis=[Spectral.Basis.Legendre],
                            quadrature=[Spectral.Quadrature.GaussLobatto],
                        )
                    ],
                    serialized_observation_functions_of_time=serialized_fots,
                )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_functions_of_time_from_volume(self):
        expected_rot_fot = [
            [0.5, 0.5, 0.5, 0.5],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
        expected_exp_fot = [0.0, 0.0, 0.0]
        expected_exp_outer_bdry_fot = [0.0, 0.0, 0.0]
        expected_translation_fot = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]

        fot_dict = functions_of_time_from_volume(
            str(self.volume_data), "ForContinuation", 6.0
        )

        self.assertEqual(fot_dict["Expansion"], expected_exp_fot)
        self.assertEqual(
            fot_dict["ExpansionOuterBoundary"], expected_exp_outer_bdry_fot
        )
        self.assertEqual(fot_dict["Rotation"], expected_rot_fot)
        self.assertEqual(fot_dict["Translation"], expected_translation_fot)


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
