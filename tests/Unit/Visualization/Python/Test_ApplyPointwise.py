# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.Visualization.ApplyPointwise import (snake_case_to_camel_case,
                                                  Kernel, apply_pointwise,
                                                  apply_pointwise_command)

import numpy as np
import numpy.testing as npt
import os
import shutil
import spectre.IO.H5 as spectre_h5
import unittest
from click.testing import CliRunner
from spectre.Informer import unit_test_src_path, unit_test_build_path
from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import Scalar, tnsr


def psi_squared(psi: Scalar[DataVector]) -> Scalar[DataVector]:
    return Scalar[DataVector](np.array(psi)**2)


def coordinate_radius(
        inertial_coordinates: tnsr.I[DataVector, 3]) -> Scalar[DataVector]:
    return Scalar[DataVector](np.expand_dims(
        np.linalg.norm(np.array(inertial_coordinates), axis=0), 0))


class TestApplyPointwise(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(unit_test_build_path(),
                                     'Visualization/ApplyPointwise')
        self.h5_filename = os.path.join(self.test_dir, "Test.h5")
        os.makedirs(self.test_dir, exist_ok=True)
        shutil.copyfile(
            os.path.join(unit_test_src_path(),
                         'Visualization/Python/VolTestData0.h5'),
            self.h5_filename)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_snake_case_to_camel_case(self):
        self.assertEqual(snake_case_to_camel_case("hello_world"), "HelloWorld")

    def test_apply_pointwise(self):
        open_h5_files = [spectre_h5.H5File(self.h5_filename, "a")]
        open_volfiles = [
            h5file.get_vol("/element_data") for h5file in open_h5_files
        ]

        kernels = [Kernel(psi_squared), Kernel(coordinate_radius)]

        apply_pointwise(volfiles=open_volfiles, kernels=kernels)

        obs_id = open_volfiles[0].list_observation_ids()[0]
        result_psisq = open_volfiles[0].get_tensor_component(
            obs_id, "PsiSquared").data
        result_radius = open_volfiles[0].get_tensor_component(
            obs_id, "CoordinateRadius").data
        psi = open_volfiles[0].get_tensor_component(obs_id, "Psi").data
        x, y, z = [
            np.array(open_volfiles[0].get_tensor_component(
                obs_id, "InertialCoordinates" + xyz).data)
            for xyz in ["_x", "_y", "_z"]
        ]
        npt.assert_allclose(np.array(result_psisq), np.array(psi)**2)
        npt.assert_allclose(np.array(result_radius),
                            np.sqrt(x**2 + y**2 + z**2))


if __name__ == '__main__':
    unittest.main(verbosity=2)
