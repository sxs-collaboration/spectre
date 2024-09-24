# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import unittest

import numpy as np
from click.testing import CliRunner

import spectre.IO.H5 as spectre_h5
from spectre import Informer
from spectre.DataStructures import DataVector
from spectre.IO.H5 import ElementVolumeData, TensorComponent, combine_h5
from spectre.IO.H5.CombineH5 import combine_h5_command
from spectre.Spectral import Basis, Quadrature


class TestCombineH5(unittest.TestCase):
    # Test Fixtures
    def setUp(self):
        # The tests in this class combine 2 HDF5 files to generate one,
        # using the Combine_H5 functionality.
        self.file_name1 = os.path.join(
            Informer.unit_test_build_path(), "IO/TestVolumeData0.h5"
        )
        self.file_name2 = os.path.join(
            Informer.unit_test_build_path(), "IO/TestVolumeData1.h5"
        )
        self.file_names = [self.file_name1, self.file_name2]
        self.subfile_name = "/element_data"

        self.output_file = os.path.join(
            Informer.unit_test_build_path(), "IO/TestOutput.h5"
        )

        if os.path.isfile(self.file_name1):
            os.remove(self.file_name1)
        if os.path.isfile(self.file_name2):
            os.remove(self.file_name2)
        if os.path.isfile(self.output_file):
            os.remove(self.output_file)

        # Initializing attributes
        grid_names1 = ["[B0(L0I0,L0I0,L1I0)]"]
        grid_names2 = ["[B1(L1I0,L0I0,L0I0)]"]
        observation_values = {0: 7.0, 1: 1.3}
        basis = Basis.Legendre
        quad = Quadrature.Gauss
        self.observation_ids = [0, 1]

        # Writing ElementVolume data and TensorComponent Data to first file
        self.h5_file1 = spectre_h5.H5File(file_name=self.file_name1, mode="a")
        self.tensor_component_data1 = np.array(
            [
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
            ]
        )
        self.h5_file1.insert_vol(self.subfile_name, version=0)
        self.h5_file1.close_current_object()
        self.vol_file1 = self.h5_file1.get_vol(self.subfile_name)

        self.element_vol_data_file_1 = [
            ElementVolumeData(
                element_name=grid_names1[0],
                components=[
                    TensorComponent(
                        "field_1", DataVector(self.tensor_component_data1[i])
                    ),
                    TensorComponent(
                        "field_2",
                        DataVector(self.tensor_component_data1[i]),
                    ),
                ],
                extents=3 * [2],
                basis=3 * [basis],
                quadrature=3 * [quad],
            )
            for i, observation_id in enumerate(self.observation_ids)
        ]

        # Write extents and tensor volume data to the volfile
        for i, observation_id in enumerate(self.observation_ids):
            self.vol_file1.write_volume_data(
                observation_id,
                observation_values[observation_id],
                [
                    self.element_vol_data_file_1[i],
                ],
            )

        # Store initial connectivity data
        self.initial_h5_connectivity = self.vol_file1.get_tensor_component(
            self.observation_ids[0], "connectivity"
        ).data

        self.h5_file1.close()

        # Writing ElementVolume data and TensorComponent Data to second file
        self.h5_file2 = spectre_h5.H5File(file_name=self.file_name2, mode="a")
        self.tensor_component_data2 = np.array(
            [
                [0.1, 0.12, 0.22, 0.32, 0.42, 0.52, 0.62, 0.72],
                [0.011, 0.021, 0.031, 0.041, 0.051, 0.061, 0.079, 0.089],
            ]
        )
        self.h5_file2.insert_vol(self.subfile_name, version=0)
        self.h5_file2.close_current_object()
        self.vol_file2 = self.h5_file2.get_vol(self.subfile_name)

        self.element_vol_data_file_2 = [
            ElementVolumeData(
                element_name=grid_names2[0],
                components=[
                    TensorComponent(
                        "field_1", DataVector(self.tensor_component_data2[i])
                    ),
                    TensorComponent(
                        "field_2",
                        DataVector(self.tensor_component_data2[i]),
                    ),
                ],
                extents=3 * [2],
                basis=3 * [basis],
                quadrature=3 * [quad],
            )
            for i, observation_id in enumerate(self.observation_ids)
        ]

        # Write extents and tensor volume data to the volfile
        for i, observation_id in enumerate(self.observation_ids):
            self.vol_file2.write_volume_data(
                observation_id,
                observation_values[observation_id],
                [
                    self.element_vol_data_file_2[i],
                ],
            )
        self.h5_file2.close()

    def tearDown(self):
        self.h5_file1.close()
        self.h5_file2.close()
        if os.path.isfile(self.file_name1):
            os.remove(self.file_name1)
        if os.path.isfile(self.file_name2):
            os.remove(self.file_name2)
        if os.path.isfile(self.output_file):
            os.remove(self.output_file)

    def test_combine_h5(self):
        # Run the combine_h5 command and check if any feature (for eg.
        # connectivity length has increased due to combining two files)

        combine_h5(
            self.file_names,
            self.subfile_name,
            self.output_file,
            None,
            None,
            False,
        )
        h5_output = spectre_h5.H5File(file_name=self.output_file, mode="r")
        output_vol = h5_output.get_vol(self.subfile_name)

        # Test observation ids

        actual_obs_ids = output_vol.list_observation_ids()
        actual_obs_ids.sort()

        expected_obs_ids = [0, 1]

        self.assertEqual(actual_obs_ids, expected_obs_ids)

        # Test observation values

        obs_id = actual_obs_ids[0]

        actual_obs_value = output_vol.get_observation_value(obs_id)
        expected_obs_value = 7.0

        self.assertEqual(actual_obs_value, expected_obs_value)

        # Test tensor components

        actual_tensor_component_names = output_vol.list_tensor_components(
            obs_id
        )

        expected_tensor_component_names = ["field_1", "field_2"]

        self.assertEqual(
            actual_tensor_component_names, expected_tensor_component_names
        )

        expected_tensor_components = np.array(
            [
                (
                    0,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.1,
                    0.12,
                    0.22,
                    0.32,
                    0.42,
                    0.52,
                    0.62,
                    0.72,
                ),
                (
                    0.01,
                    0.02,
                    0.03,
                    0.04,
                    0.05,
                    0.06,
                    0.07,
                    0.08,
                    0.011,
                    0.021,
                    0.031,
                    0.041,
                    0.051,
                    0.061,
                    0.079,
                    0.089,
                ),
            ]
        )
        for i in range(len(expected_tensor_components)):
            value = (
                output_vol.get_tensor_component(
                    i, actual_tensor_component_names[i]
                ).data
                == expected_tensor_components[i]
            )
            self.assertEqual(value, True)

    def test_cli(self):
        # Checks if the CLI for CombineH5 runs properly
        runner = CliRunner()
        result = runner.invoke(
            combine_h5_command,
            [
                "vol",
                *self.file_names,
                "-d",
                self.subfile_name,
                "-o",
                self.output_file,
                "--check-src",
            ],
            catch_exceptions=False,
        )

        h5_output = spectre_h5.H5File(file_name=self.output_file, mode="r")
        output_vol = h5_output.get_vol(self.subfile_name)

        # Extracts the connectivity data from the volume file
        # If length of final connectivity is more, combine_h5
        # has successfully merged the two HDF5 files.

        final_h5_connectivity = output_vol.get_tensor_component(
            self.observation_ids[0], "connectivity"
        ).data

        assert len(self.initial_h5_connectivity) < len(final_h5_connectivity)
        self.assertEqual(result.exit_code, 0, result.output)

    def test_cli2(self):
        # Checks that if no subfile name is given, the CLI prints
        # available subfiles correctly
        runner = CliRunner()
        result = runner.invoke(
            combine_h5_command,
            [
                *self.file_names,
                "-o",
                self.output_file,
                "--check-src",
            ],
            catch_exceptions=False,
        )
        assert result.output is not None


if __name__ == "__main__":
    unittest.main(verbosity=2)
