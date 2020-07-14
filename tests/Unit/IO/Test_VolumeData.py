# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre import DataStructures
import spectre.IO.H5 as spectre_h5
from spectre import Informer
import unittest
import numpy as np
import os
import numpy.testing as npt


class TestIOH5VolumeData(unittest.TestCase):
    # Test Fixtures
    def setUp(self):
        # The First tests involve inserting vol files, the h5 file for
        # these will be deleted and recreated for each test as needed.
        self.file_name_w = os.path.join(Informer.unit_test_path(),
                                        "IO/pythontest.h5")
        # The other tests require external data, so we use the file
        # `VolTestData.h5` which contains data output from the spectre
        # EvolvePlaneWave3D executable.  However, the file does not contain the
        # source tree (for file size reasons), so it should not be used as an
        # example output for any other reason than testing VolumeData.
        self.file_name_r = os.path.join(Informer.unit_test_path(),
                                        "IO/VolTestData.h5")
        if os.path.isfile(self.file_name_w):
            os.remove(self.file_name_w)

    def tearDown(self):
        if os.path.isfile(self.file_name_w):
            os.remove(self.file_name_w)

    # Testing the VolumeData Insert Function
    def test_insert_vol(self):
        h5_file = spectre_h5.H5File(file_name=self.file_name_w,
                                    append_to_file=True)
        h5_file.insert_vol(path="/element_data", version=0)
        vol_file = h5_file.get_vol(path="/element_data")
        self.assertEqual(vol_file.get_version(), 0)
        h5_file.close()

    # Test the header was generated correctly
    def test_vol_get_header(self):
        h5_file = spectre_h5.H5File(file_name=self.file_name_w,
                                    append_to_file=True)
        h5_file.insert_vol(path="/element_data", version=0)
        vol_file = h5_file.get_vol(path="/element_data")
        self.assertEqual(vol_file.get_header()[0:20], "#\n# File created on ")
        h5_file.close()

    # The other tests require external data, so they use the included h5 file
    # `VolTestData.h5` which contains spectre output data (see above).
    # Test the observation ids and values are correctly retrived
    def test_observation_id(self):
        h5_file = spectre_h5.H5File(file_name=self.file_name_r,
                                    append_to_file=True)
        vol_file = h5_file.get_vol(path="/element_data")
        obs_ids = set(vol_file.list_observation_ids())
        expected_obs_ids = set([16436106908031328247, 17615288952477351885])
        expected_obs_values = {
            16436106908031328247: 0.01,
            17615288952477351885: 0.00
        }
        self.assertEqual(obs_ids, expected_obs_ids)
        for obs_id in expected_obs_ids:
            self.assertEqual(
                vol_file.get_observation_value(observation_id=obs_id),
                expected_obs_values[obs_id])
        h5_file.close()

    # Test to make sure information about the computation elements was found
    def test_grids(self):
        h5_file = spectre_h5.H5File(file_name=self.file_name_r,
                                    append_to_file=True)
        vol_file = h5_file.get_vol("/element_data")
        obs_id = vol_file.list_observation_ids()[0]
        grid_names = vol_file.get_grid_names(observation_id=obs_id)
        expected_grid_names = ['[B0,(L0I0,L0I0,L0I0)]']
        self.assertEqual(grid_names, expected_grid_names)
        extents = vol_file.get_extents(observation_id=obs_id)
        expected_extents = [[2, 2, 2]]
        self.assertEqual(extents, expected_extents)
        h5_file.close()

    # Test that the tensor components, and tensor data  are retrieved correctly
    def test_tensor_components(self):
        h5_file = spectre_h5.H5File(file_name=self.file_name_r,
                                    append_to_file=True)
        vol_file = h5_file.get_vol(path="/element_data")
        obs_id = vol_file.list_observation_ids()[0]
        tensor_comps = set(
            vol_file.list_tensor_components(observation_id=obs_id))
        expected_tensor_comps = set([
            'Psi', 'Error(Psi)', 'InertialCoordinates_x',
            'InertialCoordinates_y', 'InertialCoordinates_z'
        ])
        self.assertEqual(tensor_comps, expected_tensor_comps)
        expected_Psi_tensor_data = np.array([
            -0.0173205080756888, -0.0173205080756888, -0.0173205080756888,
            -0.0173205080756888, -0.0173205080756888, -0.0173205080756888,
            -0.0173205080756888, -0.0173205080756888
        ])
        expected_Error_tensor_data = np.array([
            -8.66012413502926e-07, -8.66012413502926e-07,
            -8.66012413502926e-07, -8.66012413502926e-07,
            -8.66012413502926e-07, -8.66012413502926e-07,
            -8.66012413502926e-07, -8.66012413502926e-07
        ])
        expected_xcoord_tensor_data = np.array([
            0.0, 6.28318530717959, 0.0, 6.28318530717959, 0.0,
            6.28318530717959, 0.0, 6.28318530717959
        ])
        expected_ycoord_tensor_data = np.array([
            0.0, 0.0, 6.28318530717959, 6.28318530717959, 0.0, 0.0,
            6.28318530717959, 6.28318530717959
        ])
        expected_zcoord_tensor_data = np.array([
            0.0, 0.0, 0.0, 0.0, 6.28318530717959, 6.28318530717959,
            6.28318530717959, 6.28318530717959
        ])
        # Checking whether two numpy arrays are "almost equal" is easy, so
        # we convert everything to numpy arrays for comparison.
        Psi_tensor_data = np.asarray(
            vol_file.get_tensor_component(observation_id=obs_id,
                                          tensor_component='Psi'))
        npt.assert_array_almost_equal(Psi_tensor_data,
                                      expected_Psi_tensor_data)
        Error_tensor_data = np.asarray(
            vol_file.get_tensor_component(observation_id=obs_id,
                                          tensor_component='Error(Psi)'))
        npt.assert_array_almost_equal(Error_tensor_data,
                                      expected_Error_tensor_data)
        xcoord_tensor_data = np.asarray(
            vol_file.get_tensor_component(
                observation_id=obs_id,
                tensor_component='InertialCoordinates_x'))
        npt.assert_array_almost_equal(xcoord_tensor_data,
                                      expected_xcoord_tensor_data)
        ycoord_tensor_data = np.asarray(
            vol_file.get_tensor_component(
                observation_id=obs_id,
                tensor_component='InertialCoordinates_y'))
        npt.assert_array_almost_equal(ycoord_tensor_data,
                                      expected_ycoord_tensor_data)
        zcoord_tensor_data = np.asarray(
            vol_file.get_tensor_component(
                observation_id=obs_id,
                tensor_component='InertialCoordinates_z'))
        npt.assert_array_almost_equal(zcoord_tensor_data,
                                      expected_zcoord_tensor_data)
        h5_file.close()

    def test_offset_and_length_for_grid(self):
        h5_file = spectre_h5.H5File(file_name=self.file_name_r,
                                    append_to_file=True)
        vol_file = h5_file.get_vol(path="/element_data")
        obs_id = vol_file.list_observation_ids()[0]
        all_grid_names = vol_file.get_grid_names(observation_id=obs_id)
        all_extents = vol_file.get_extents(observation_id=obs_id)
        h5_file.close()

        self.assertEqual(
            spectre_h5.offset_and_length_for_grid(
                grid_name='[B0,(L0I0,L0I0,L0I0)]',
                all_grid_names=all_grid_names,
                all_extents=all_extents), (0, 8))


if __name__ == '__main__':
    unittest.main(verbosity=2)
