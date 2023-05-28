# Distributed under the MIT License.
# See LICENSE.txt for details.

import inspect
import os
import shutil
import unittest

import numpy as np
import numpy.testing as npt
from click.testing import CliRunner

import spectre.IO.H5 as spectre_h5
from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import InverseJacobian, Scalar, tnsr
from spectre.Informer import unit_test_build_path, unit_test_src_path
from spectre.NumericalAlgorithms.LinearOperators import partial_derivative
from spectre.PointwiseFunctions.Punctures import adm_mass_integrand
from spectre.Spectral import Mesh
from spectre.Visualization.TransformVolumeData import (
    Kernel,
    parse_pybind11_signatures,
    snake_case_to_camel_case,
    transform_volume_data,
    transform_volume_data_command,
)


def adm_mass_integrand_signature(
    field: Scalar[DataVector],
    alpha: Scalar[DataVector],
    beta: Scalar[DataVector],
) -> Scalar[DataVector]:
    pass


def psi_squared(psi: Scalar[DataVector]) -> Scalar[DataVector]:
    return Scalar[DataVector](np.array(psi) ** 2)


def coordinate_radius(inertial_coordinates: tnsr.I[DataVector, 3]):
    # Return a Numpy array
    return np.linalg.norm(np.array(inertial_coordinates), axis=0)


def deriv_coords(
    inertial_coords: tnsr.I[DataVector, 3],
    mesh: Mesh[3],
    inv_jacobian: InverseJacobian[DataVector, 3],
) -> tnsr.iJ[DataVector, 3]:
    # This should be delta_ij
    return partial_derivative(inertial_coords, mesh, inv_jacobian)


def sinusoid(x: tnsr.I[DataVector, 3]) -> Scalar[DataVector]:
    # The integral over [2 pi]^3 of this integrand is 4**3=64
    return Scalar[DataVector](
        np.expand_dims(np.prod(np.sin(0.5 * np.array(x)), axis=0), axis=0)
    )


def square_component(component: DataVector):
    # Return a DataVector
    return component**2


def abs_and_max(component: DataVector):
    # Return multiple datasets
    return {
        "Abs": component.abs(),
        # Single number, should get expanded over the volume as constant scalar
        "Max": component.max(),
    }


class TestTransformVolumeData(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(
            unit_test_build_path(), "Visualization/TransformVolumeData"
        )
        self.h5_filename = os.path.join(self.test_dir, "Test.h5")
        os.makedirs(self.test_dir, exist_ok=True)
        shutil.copyfile(
            os.path.join(
                unit_test_src_path(), "Visualization/Python/VolTestData0.h5"
            ),
            self.h5_filename,
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_parse_pybind11_signatures(self):
        self.assertEqual(
            list(parse_pybind11_signatures(adm_mass_integrand)),
            [inspect.signature(adm_mass_integrand_signature)],
        )

    def test_snake_case_to_camel_case(self):
        self.assertEqual(snake_case_to_camel_case("hello_world"), "HelloWorld")

    def test_transform_volume_data(self):
        open_h5_files = [spectre_h5.H5File(self.h5_filename, "a")]
        open_volfiles = [
            h5file.get_vol("/element_data") for h5file in open_h5_files
        ]

        kernels = [
            Kernel(psi_squared),
            Kernel(coordinate_radius),
            Kernel(
                coordinate_radius,
                elementwise=True,
                output_name="CoordinateRadiusElementwise",
            ),
            Kernel(deriv_coords),
            Kernel(
                square_component,
                map_input_names={"component": "InertialCoordinates_x"},
            ),
            Kernel(abs_and_max, map_input_names={"component": "Psi"}),
        ]

        transform_volume_data(volfiles=open_volfiles, kernels=kernels)

        obs_id = open_volfiles[0].list_observation_ids()[0]
        result_psisq = (
            open_volfiles[0].get_tensor_component(obs_id, "PsiSquared").data
        )
        result_radius = (
            open_volfiles[0]
            .get_tensor_component(obs_id, "CoordinateRadius")
            .data
        )
        result_radius_elementwise = (
            open_volfiles[0]
            .get_tensor_component(obs_id, "CoordinateRadiusElementwise")
            .data
        )
        psi = open_volfiles[0].get_tensor_component(obs_id, "Psi").data
        x, y, z = [
            np.array(
                open_volfiles[0]
                .get_tensor_component(obs_id, "InertialCoordinates" + xyz)
                .data
            )
            for xyz in ["_x", "_y", "_z"]
        ]
        radius = np.sqrt(x**2 + y**2 + z**2)
        npt.assert_allclose(np.array(result_psisq), np.array(psi) ** 2)
        npt.assert_allclose(np.array(result_radius), radius)
        npt.assert_allclose(np.array(result_radius_elementwise), radius)
        for i in range(3):
            result_deriv_coords = (
                open_volfiles[0]
                .get_tensor_component(
                    obs_id, "DerivCoords_" + "xyz"[i] + "xyz"[i]
                )
                .data
            )
            npt.assert_allclose(result_deriv_coords, 1.0)
            for j in range(i):
                result_deriv_coords = (
                    open_volfiles[0]
                    .get_tensor_component(
                        obs_id, "DerivCoords_" + "xyz"[i] + "xyz"[j]
                    )
                    .data
                )
                npt.assert_allclose(result_deriv_coords, 0.0, atol=1e-14)
        result_square_component = (
            open_volfiles[0]
            .get_tensor_component(obs_id, "SquareComponent")
            .data
        )
        npt.assert_allclose(np.array(result_square_component), x**2)
        result_abs = open_volfiles[0].get_tensor_component(obs_id, "Abs").data
        npt.assert_allclose(np.array(result_abs), np.abs(np.array(psi)))
        result_max = open_volfiles[0].get_tensor_component(obs_id, "Max").data
        npt.assert_allclose(
            np.array(result_max), np.ones(len(radius)) * np.max(np.array(psi))
        )

    def test_integrate(self):
        open_h5_files = [spectre_h5.H5File(self.h5_filename, "a")]
        open_volfiles = [
            h5file.get_vol("/element_data") for h5file in open_h5_files
        ]

        kernels = [
            Kernel(sinusoid),
        ]

        integrals = transform_volume_data(
            volfiles=open_volfiles, kernels=kernels, integrate=True
        )

        npt.assert_allclose(integrals["Volume"], (2 * np.pi) ** 3)
        # The domain has pretty low resolution so the integral is not
        # particularly precise
        npt.assert_allclose(integrals["Sinusoid"], 64.0, rtol=1e-2)

    def test_cli(self):
        runner = CliRunner()
        cli_flags = [
            self.h5_filename,
            "-d",
            "element_data.vol",
            "-e",
            __file__,
        ]
        result = runner.invoke(
            transform_volume_data_command,
            cli_flags
            + [
                "-k",
                "psi_squared",
                "-k",
                "coordinate_radius",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)
        with spectre_h5.H5File(self.h5_filename, "r") as open_h5_file:
            volfile = open_h5_file.get_vol("/element_data")
            obs_id = volfile.list_observation_ids()[0]
            psi = volfile.get_tensor_component(obs_id, "Psi").data
            result_psisq = volfile.get_tensor_component(
                obs_id, "PsiSquared"
            ).data
            npt.assert_allclose(np.array(result_psisq), np.array(psi) ** 2)
            x, y, z = [
                np.array(
                    volfile.get_tensor_component(
                        obs_id, "InertialCoordinates" + xyz
                    ).data
                )
                for xyz in ["_x", "_y", "_z"]
            ]
            radius = np.sqrt(x**2 + y**2 + z**2)
            result_radius = volfile.get_tensor_component(
                obs_id, "CoordinateRadius"
            ).data
            npt.assert_allclose(np.array(result_radius), radius)

        # Test integrals
        result = runner.invoke(
            transform_volume_data_command,
            cli_flags
            + [
                "-k",
                "sinusoid",
                "--integrate",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("63.88", result.output)

        output_filename = os.path.join(self.test_dir, "integrals.h5")
        result = runner.invoke(
            transform_volume_data_command,
            cli_flags
            + [
                "-k",
                "sinusoid",
                "--integrate",
                "--output",
                output_filename,
                "--output-subfile",
                "integrals",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)
        with spectre_h5.H5File(output_filename, "r") as open_h5_file:
            datfile = open_h5_file.get_dat("/integrals")
            self.assertEqual(
                datfile.get_legend(), ["Time", "Volume", "Sinusoid"]
            )
            npt.assert_allclose(
                datfile.get_data(),
                [[0.04, (2.0 * np.pi) ** 3, 64.0]],
                rtol=1e-2,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
