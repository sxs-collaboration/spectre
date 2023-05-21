# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import unittest

import numpy as np
import numpy.testing as npt

import spectre.IO.H5 as spectre_h5
from spectre import Informer
from spectre.DataStructures import DataVector
from spectre.IO.H5 import ElementVolumeData, TensorComponent
from spectre.Spectral import Basis, Quadrature


class TestVolumeDataWriting(unittest.TestCase):
    # Test Fixtures
    def setUp(self):
        # The tests in this class involve inserting vol files, the h5 file
        # will be deleted and recreated for each test
        self.file_name = os.path.join(
            Informer.unit_test_build_path(), "IO/TestVolumeDataWriting.h5"
        )
        if os.path.isfile(self.file_name):
            os.remove(self.file_name)
        self.h5_file = spectre_h5.H5File(file_name=self.file_name, mode="a")

    def tearDown(self):
        self.h5_file.close()
        if os.path.isfile(self.file_name):
            os.remove(self.file_name)

    # Testing the VolumeData Insert Function
    def test_insert_vol(self):
        self.h5_file.insert_vol(path="/element_data", version=0)
        self.h5_file.close_current_object()
        vol_file = self.h5_file.get_vol(path="/element_data")
        self.assertEqual(vol_file.get_version(), 0)

    # Test the header was generated correctly
    def test_vol_get_header(self):
        self.h5_file.insert_vol(path="/element_data", version=0)
        self.h5_file.close_current_object()
        vol_file = self.h5_file.get_vol(path="/element_data")
        self.assertEqual(vol_file.get_header()[0:20], "#\n# File created on ")


class TestVolumeData(unittest.TestCase):
    # Test Fixtures
    def setUp(self):
        # The tests in this class use a volume data file written using
        # the write_volume_data() function
        self.file_name = os.path.join(
            Informer.unit_test_build_path(), "IO/TestVolumeData.h5"
        )

        if os.path.isfile(self.file_name):
            os.remove(self.file_name)

        self.h5_file = spectre_h5.H5File(file_name=self.file_name, mode="a")
        self.tensor_component_data = np.random.rand(4, 8)
        observation_ids = [0, 1]
        observation_values = {0: 7.0, 1: 1.3}
        grid_names = ["grid_1", "grid_2"]
        basis = Basis.Legendre
        quad = Quadrature.Gauss

        # Insert .vol file to h5 file
        self.h5_file.insert_vol("/element_data", version=0)
        self.h5_file.close_current_object()
        self.vol_file = self.h5_file.get_vol(path="/element_data")

        # Set TensorComponent and ElementVolumeData to be written

        self.element_vol_data_grid_1 = [
            ElementVolumeData(
                element_name=grid_names[0],
                components=[
                    TensorComponent(
                        "field_1", DataVector(self.tensor_component_data[2 * i])
                    ),
                    TensorComponent(
                        "field_2",
                        DataVector(self.tensor_component_data[2 * i + 1]),
                    ),
                ],
                extents=3 * [2],
                basis=3 * [basis],
                quadrature=3 * [quad],
            )
            for i, observation_id in enumerate(observation_ids)
        ]

        self.element_vol_data_grid_2 = [
            ElementVolumeData(
                element_name=grid_names[1],
                components=[
                    TensorComponent(
                        "field_1",
                        DataVector(self.tensor_component_data[2 * i + 1]),
                    ),
                    TensorComponent(
                        "field_2", DataVector(self.tensor_component_data[2 * i])
                    ),
                ],
                extents=3 * [2],
                basis=3 * [basis],
                quadrature=3 * [quad],
            )
            for i, observation_id in enumerate(observation_ids)
        ]

        # Write extents and tensor volume data to volfile

        for i, observation_id in enumerate(observation_ids):
            self.vol_file.write_volume_data(
                observation_id,
                observation_values[observation_id],
                [
                    self.element_vol_data_grid_1[i],
                    self.element_vol_data_grid_2[i],
                ],
            )

    def tearDown(self):
        self.h5_file.close()
        if os.path.isfile(self.file_name):
            os.remove(self.file_name)

    # Test that observation ids and values are retrieved correctly
    def test_observation_id(self):
        # Test observation Ids
        obs_ids = set(self.vol_file.list_observation_ids())
        expected_obs_ids = set([0, 1])
        self.assertEqual(obs_ids, expected_obs_ids)
        # Test observation values
        expected_obs_values = {0: 7.0, 1: 1.3}
        for obs_id in expected_obs_ids:
            self.assertEqual(
                self.vol_file.get_observation_value(observation_id=obs_id),
                expected_obs_values[obs_id],
            )

    # Test to make sure information about the computation elements was found
    def test_grids(self):
        obs_id = self.vol_file.list_observation_ids()[0]
        # Test grid names
        grid_names = self.vol_file.get_grid_names(observation_id=obs_id)
        expected_grid_names = ["grid_1", "grid_2"]
        self.assertEqual(grid_names, expected_grid_names)
        # Test extents
        extents = self.vol_file.get_extents(observation_id=obs_id)
        expected_extents = [[2, 2, 2], [2, 2, 2]]
        self.assertEqual(extents, expected_extents)
        # Test bases
        bases = self.vol_file.get_bases(obs_id)
        expected_bases = 2 * [3 * [Basis.Legendre]]
        self.assertEqual(bases, expected_bases)
        # Test quadratures
        quadratures = self.vol_file.get_quadratures(obs_id)
        expected_quadratures = 2 * [3 * [Quadrature.Gauss]]
        self.assertEqual(quadratures, expected_quadratures)
        # Test dimension
        dim = self.vol_file.get_dimension()
        self.assertEqual(dim, 3)

    # Test that the tensor components, and tensor data  are retrieved correctly
    def test_tensor_components(self):
        obs_id = 0
        # Test tensor component names
        tensor_component_names = set(
            self.vol_file.list_tensor_components(observation_id=obs_id)
        )
        expected_tensor_component_names = ["field_1", "field_2"]
        self.assertEqual(
            tensor_component_names, set(expected_tensor_component_names)
        )
        # Test tensor component data at specified obs_id
        for i, expected_tensor_component_data in enumerate(
            self.tensor_component_data[:2]
        ):
            npt.assert_almost_equal(
                np.asarray(
                    self.vol_file.get_tensor_component(
                        observation_id=obs_id,
                        tensor_component=expected_tensor_component_names[i],
                    ).data
                )[0:8],
                expected_tensor_component_data,
            )

    def test_get_data_by_element(self):
        obs_id = 0
        volume_data = self.vol_file.get_data_by_element(None, None, None)
        self.assertEqual(len(volume_data), 2)

        # Check both grids at first observation time
        self.assertEqual(volume_data[0][0], 1)
        self.assertEqual(volume_data[0][1], 1.3)
        self.assertEqual(volume_data[0][2][0].element_name, "grid_1")
        self.assertEqual(
            volume_data[0][2][0].basis, self.element_vol_data_grid_1[1].basis
        )
        self.assertEqual(
            volume_data[0][2][0].quadrature,
            self.element_vol_data_grid_1[1].quadrature,
        )
        self.assertEqual(
            volume_data[0][2][0].extents,
            self.element_vol_data_grid_1[1].extents,
        )
        self.assertEqual(
            volume_data[0][2][0].tensor_components,
            self.element_vol_data_grid_1[1].tensor_components,
        )

        self.assertEqual(volume_data[0][2][1].element_name, "grid_2")
        self.assertEqual(
            volume_data[0][2][1].basis, self.element_vol_data_grid_2[1].basis
        )
        self.assertEqual(
            volume_data[0][2][1].quadrature,
            self.element_vol_data_grid_2[1].quadrature,
        )
        self.assertEqual(
            volume_data[0][2][1].extents,
            self.element_vol_data_grid_2[1].extents,
        )
        self.assertEqual(
            volume_data[0][2][1].tensor_components,
            self.element_vol_data_grid_2[1].tensor_components,
        )

        # Check both grids at second observation time
        self.assertEqual(volume_data[1][0], 0)
        self.assertEqual(volume_data[1][1], 7.0)
        self.assertEqual(volume_data[1][2][0].element_name, "grid_1")
        self.assertEqual(
            volume_data[1][2][0].basis, self.element_vol_data_grid_1[1].basis
        )
        self.assertEqual(
            volume_data[1][2][0].quadrature,
            self.element_vol_data_grid_1[1].quadrature,
        )
        self.assertEqual(
            volume_data[1][2][0].extents,
            self.element_vol_data_grid_1[1].extents,
        )
        self.assertEqual(
            volume_data[1][2][0].tensor_components,
            self.element_vol_data_grid_1[0].tensor_components,
        )

        self.assertEqual(volume_data[1][2][1].element_name, "grid_2")
        self.assertEqual(
            volume_data[1][2][1].basis, self.element_vol_data_grid_2[1].basis
        )
        self.assertEqual(
            volume_data[1][2][1].quadrature,
            self.element_vol_data_grid_2[1].quadrature,
        )
        self.assertEqual(
            volume_data[1][2][1].extents,
            self.element_vol_data_grid_2[1].extents,
        )
        self.assertEqual(
            volume_data[1][2][1].tensor_components,
            self.element_vol_data_grid_2[0].tensor_components,
        )

    # Test that the offset and length for certain grid is retrieved correctly
    def test_offset_and_length_for_grid(self):
        obs_id = self.vol_file.list_observation_ids()[0]
        all_grid_names = self.vol_file.get_grid_names(observation_id=obs_id)
        all_extents = self.vol_file.get_extents(observation_id=obs_id)
        self.assertEqual(
            spectre_h5.offset_and_length_for_grid(
                grid_name="grid_1",
                all_grid_names=all_grid_names,
                all_extents=all_extents,
            ),
            (0, 8),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
