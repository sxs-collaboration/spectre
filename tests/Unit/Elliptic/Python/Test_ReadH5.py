# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import shutil
import unittest

import numpy as np
import numpy.testing as npt

import spectre.IO.H5 as spectre_h5
from spectre.Domain import ElementId
from spectre.Elliptic.ReadH5 import read_matrix
from spectre.Informer import unit_test_build_path
from spectre.Spectral import Basis, Mesh, Quadrature


class TestReadH5(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(
            unit_test_build_path(), "Elliptic/Python/ReadH5"
        )
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_read_matrix(self):
        filename = os.path.join(self.test_dir, "TestMatrix.h5")
        subfile_name = "/Matrix"
        num_vars = 2
        num_points = 3
        num_elements = 2
        mesh = Mesh[1](num_points, Basis.Legendre, Quadrature.GaussLobatto)
        size = num_vars * num_points * num_elements
        expected_matrix = np.random.rand(size, size)
        # Write matrix to file
        with spectre_h5.H5File(filename, "w") as h5file:
            volfile = h5file.insert_vol(subfile_name, version=0)
            for col in range(size):
                volume_data = [
                    spectre_h5.ElementVolumeData(
                        element_id=ElementId[1](0),
                        components=[
                            spectre_h5.TensorComponent(
                                "Variable_0", expected_matrix[:3, col]
                            ),
                            spectre_h5.TensorComponent(
                                "Variable_1", expected_matrix[3:6, col]
                            ),
                        ],
                        mesh=mesh,
                    ),
                    spectre_h5.ElementVolumeData(
                        element_id=ElementId[1](1),
                        components=[
                            spectre_h5.TensorComponent(
                                "Variable_0", expected_matrix[6:9, col]
                            ),
                            spectre_h5.TensorComponent(
                                "Variable_1", expected_matrix[9:12, col]
                            ),
                        ],
                        mesh=mesh,
                    ),
                ]
                volfile.write_volume_data(
                    observation_id=col,
                    observation_value=col,
                    elements=volume_data,
                )
                npt.assert_array_equal(
                    volfile.get_tensor_component(col, "Variable_1").data,
                    np.concatenate(
                        [expected_matrix[3:6, col], expected_matrix[9:12, col]]
                    ),
                )
        # Test reading matrix
        with spectre_h5.H5File(filename, "r") as h5file:
            volfile = h5file.get_vol(subfile_name)
            matrix = read_matrix(volfile)
        npt.assert_array_equal(matrix, expected_matrix)


if __name__ == "__main__":
    unittest.main(verbosity=2)
