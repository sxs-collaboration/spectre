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
from spectre.Pipelines.Bbh.InitialData import generate_id
from spectre.Pipelines.Bbh.Inspiral import start_inspiral
from spectre.Pipelines.Bbh.Ringdown import (
    ringdown_parameters,
    start_ringdown_command,
)
from spectre.support.Logging import configure_logging


class TestInitialData(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(
            unit_test_build_path(), "support/Pipelines/Bbh/Ringdown"
        )
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.bin_dir = Path(unit_test_build_path(), "../../bin").resolve()
        generate_id(
            mass_a=0.6,
            mass_b=0.4,
            dimensionless_spin_a=[0.0, 0.0, 0.0],
            dimensionless_spin_b=[0.0, 0.0, 0.0],
            separation=20.0,
            orbital_angular_velocity=0.01,
            radial_expansion_velocity=-1.0e-5,
            refinement_level=1,
            polynomial_order=5,
            run_dir=self.test_dir / "ID",
            scheduler=None,
            submit=False,
            executable=str(self.bin_dir / "SolveXcts"),
        )
        self.id_dir = self.test_dir / "ID"
        self.horizons_filename = self.id_dir / "Horizons.h5"
        with spectre_h5.H5File(
            str(self.horizons_filename.resolve()), "a"
        ) as horizons_file:
            legend = ["Time", "ChristodoulouMass", "DimensionlessSpinMagnitude"]
            for subfile_name in ["AhA", "AhB"]:
                horizons_file.close_current_object()
                dat_file = horizons_file.try_insert_dat(subfile_name, legend, 0)
                dat_file.append([[0.0, 1.0, 0.3]])
        start_inspiral(
            id_input_file_path=self.test_dir / "ID" / "InitialData.yaml",
            refinement_level=1,
            polynomial_order=5,
            segments_dir=self.test_dir / "Inspiral",
            scheduler=None,
            submit=False,
            executable=str(self.bin_dir / "EvolveGhBinaryBlackHole"),
        )
        self.inspiral_dir = self.test_dir / "Inspiral" / "Segment_0000"
        # Making fake reduction data with simple derivative
        self.inspiral_reduction_data = self.inspiral_dir / "BbhReductions.h5"
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
            shape_coefs = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            times = [4990, 4992, 4994, 4996, 4998, 5000]
            reduction_dat = reduction_file.try_insert_dat(
                "ObservationAhC_Ylm.dat", legend, 0
            )
            for x in range(0, 5):
                reduction_dat.append(
                    [
                        [
                            times[x],
                            0.0,
                            0.0,
                            0.0,
                            2,
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

        # Making volume data for functions of time to be extracted
        rotation_fot = QuaternionFunctionOfTime(
            0.0,
            [DataVector(size=4, fill=1.0)],
            4 * [DataVector(size=3, fill=0.0)],
            math.inf,
        )
        expansion_fot = PiecewisePolynomial3(
            0.0, 4 * [DataVector(size=1, fill=1.0)], math.inf
        )
        expansion_outer_fot = PiecewisePolynomial3(
            0.0, 4 * [DataVector(size=1, fill=1.0)], math.inf
        )
        serialized_fots = serialize_functions_of_time(
            {
                "Expansion": expansion_fot,
                "ExpansionOuterBoundary": expansion_outer_fot,
                "Rotation": rotation_fot,
            }
        )
        self.inspiral_volume_data = self.inspiral_dir / "BbhVolume0.h5"
        obs_values = [4990.0, 4992.0, 4994.0, 4996.0, 4998.0, 5000.0]
        with spectre_h5.H5File(self.inspiral_volume_data, "w") as volume_file:
            volfile = volume_file.insert_vol("ForContinuation", version=0)
            for x in range(0, 5):
                volfile.write_volume_data(
                    observation_id=x,
                    observation_value=obs_values[x],
                    elements=[
                        ElementVolumeData(
                            element_name="foo",
                            components=[
                                TensorComponent(
                                    "bar",
                                    np.random.rand(3),
                                ),
                            ],
                            extents=[3],
                            basis=[Spectral.Basis.Legendre],
                            quadrature=[Spectral.Quadrature.GaussLobatto],
                        )
                    ],
                    serialized_functions_of_time=serialized_fots,
                )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_ringdown_parameters(self):
        with open(self.inspiral_dir / "Inspiral.yaml") as open_input_file:
            _, inspiral_input_file = yaml.safe_load_all(open_input_file)
        params = ringdown_parameters(
            inspiral_input_file,
            self.inspiral_dir,
            "ForContinuation",
            refinement_level=1,
            polynomial_order=5,
        )
        self.assertEqual(
            params["IdFileGlob"],
            str((self.inspiral_dir).resolve() / "BbhVolume*.h5"),
        )
        self.assertEqual(params["L"], 1)
        self.assertEqual(params["P"], 5)

    def test_cli(self):
        # Not using `CliRunner.invoke()` because it runs in an isolated
        # environment and doesn't work with MPI in the container.
        try:
            start_ringdown_command(
                [
                    str(self.inspiral_dir),
                    "--refinement-level",
                    "1",
                    "--polynomial-order",
                    "5",
                    "-O",
                    str(self.test_dir),
                    "--match-time",
                    "5000.0",
                    "--number-of-ahc-finds-for-fit",
                    "5",
                    "--settling-timescale",
                    "10.0",
                    "--path-to-output-h5",
                    str(self.test_dir / "RingdownCoefs.h5"),
                    "-E",
                    str(self.bin_dir / "EvolveGhSingleBlackHole"),
                    "--no-submit",
                ]
            )
        except SystemExit as e:
            self.assertEqual(e.code, 0)
        self.assertTrue((self.test_dir / "Segment_0000/Ringdown.yaml").exists())
        self.assertTrue(
            (self.test_dir / "RingdownCoefs.h5").exists(),
        )


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
