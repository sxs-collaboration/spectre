# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import unittest

import numpy as np

import spectre.IO.H5 as spectre_h5
from spectre import Informer
from spectre.DataStructures import DataVector
from spectre.IO.H5 import ElementVolumeData, TensorComponent, combine_h5
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

        self.output_file = os.path.join(
            Informer.unit_test_build_path(), "IO/TestOutput"
        )

        if os.path.isfile(self.file_name1):
            os.remove(self.file_name1)
        if os.path.isfile(self.file_name2):
            os.remove(self.file_name2)
        if os.path.isfile(self.output_file + "0.h5"):
            os.remove(self.output_file + "0.h5")

        # Initializing attributes
        grid_names1 = ["[B0(L0I0,L0I0,L1I0)]"]
        grid_names2 = ["[B0(L1I0,L0I0,L0I0)]"]
        observation_values = {0: 7.0, 1: 1.3}
        basis = Basis.Legendre
        quad = Quadrature.Gauss
        self.observation_ids = [0, 1]

        # Writing ElementVolume data and TensorComponent Data to first file
        self.h5_file1 = spectre_h5.H5File(file_name=self.file_name1, mode="a")
        self.tensor_component_data1 = np.array([[0.0], [0.3], [0.6]])
        self.h5_file1.insert_vol("/element_data", version=0)
        self.h5_file1.close_current_object()
        self.vol_file1 = self.h5_file1.get_vol(path="/element_data")

        self.element_vol_data_file_1 = [
            ElementVolumeData(
                element_name=grid_names1[0],
                components=[
                    TensorComponent(
                        "field_1", DataVector(self.tensor_component_data1[i])
                    ),
                    TensorComponent(
                        "field_2",
                        DataVector(self.tensor_component_data1[i + 1]),
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
        self.tensor_component_data2 = np.array([[1.0], [0.13], [0.16]])
        self.h5_file2.insert_vol("/element_data", version=0)
        self.h5_file2.close_current_object()
        self.vol_file2 = self.h5_file2.get_vol(path="/element_data")

        self.element_vol_data_file_2 = [
            ElementVolumeData(
                element_name=grid_names2[0],
                components=[
                    TensorComponent(
                        "field_1", DataVector(self.tensor_component_data2[i])
                    ),
                    TensorComponent(
                        "field_2",
                        DataVector(self.tensor_component_data2[i + 1]),
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
        if os.path.isfile(self.output_file + "0.h5"):
            os.remove(self.output_file + "0.h5")

    def test_combine_h5(self):
        # Run the combine_h5 command and check if any feature (for eg.
        # connectivity length has increased due to combining two files)

        combine_h5(self.file_name1[:-4], "element_data", self.output_file)
        h5_output = spectre_h5.H5File(
            file_name=self.output_file + "0.h5", mode="a"
        )
        output_vol = h5_output.get_vol(path="/element_data")

        # Extracts the connectivity data from the volume file
        # If length of final connectivity is more, combine_h5
        # has successfully merged the two HDF5 files.

        final_h5_connectivity = output_vol.get_tensor_component(
            self.observation_ids[0], "connectivity"
        ).data

        assert len(self.initial_h5_connectivity) < len(final_h5_connectivity)


if __name__ == "__main__":
    unittest.main(verbosity=2)
