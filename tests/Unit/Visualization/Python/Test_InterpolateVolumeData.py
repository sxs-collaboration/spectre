# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest
from spectre.Informer import unit_test_build_path
import spectre.IO.H5 as spectre_h5
from spectre.DataStructures import (DataVector, TensorComponent,
                                    ElementVolumeData)
from spectre import Spectral
from spectre.Visualization import InterpolateVolumeData
import os
import numpy as np


class TestInterpolateH5(unittest.TestCase):
    def setUp(self):
        """
        Creates two files with 2 observations each with two elements each
        with two tensor components each
        """

        self.path = unit_test_build_path()
        self.file_name = os.path.join(self.path, "interpolation_1.h5")

        try:
            os.remove(self.file_name)
        except OSError:
            pass

        file1 = spectre_h5.H5File(self.file_name, "a")
        self.volume_name = "/VolumeData"
        file1.insert_vol(self.volume_name, 0)

        # interpolated solution
        self.sol1 = np.array([
            0.00363708, 0.00683545, 0.00720423, 0.00457117, 0.00143594,
            0.00269867, 0.00284427, 0.00180472, -0.00143594, -0.00269867,
            -0.00284427, -0.00180472, -0.00363708, -0.00683545, -0.00720423,
            -0.00457117, -0.00117537, -0.00107168, -0.000212089, 0.0010019,
            -0.000464041, -0.000423106, -8.37339e-05, 0.000395554, 0.000464041,
            0.000423106, 8.37339e-05, -0.000395554, 0.00117537, 0.00107168,
            0.000212089, -0.0010019, -0.00146994, -0.00281943, -0.00264259,
            -0.00102202, -0.00058034, -0.00111313, -0.00104331, -0.000403497,
            0.00058034, 0.00111313, 0.00104331, 0.000403497, 0.00146994,
            0.00281943, 0.00264259, 0.00102202, -0.000809575, -0.00215823,
            -0.00242105, -0.00147528, -0.000319624, -0.000852079, -0.000955844,
            -0.00058245, 0.000319624, 0.000852079, 0.000955844, 0.00058245,
            0.000809575, 0.00215823, 0.00242105, 0.00147528
        ])

        self.sol2 = np.array([
            0.0258164, 0.0113765, 0.00140198, 0.000552092, 0.0292701,
            0.0168924, 0.00688523, 0.00392304, 0.0292701, 0.0168924,
            0.00688523, 0.00392304, 0.0258164, 0.0113765, 0.00140198,
            0.000552092, -0.00127999, 0.00293799, 0.0044177, 0.00246795,
            -0.00248599, 0.0020233, 0.0042722, 0.00321024, -0.00248599,
            0.0020233, 0.0042722, 0.00321024, -0.00127999, 0.00293799,
            0.0044177, 0.00246795, -0.00682319, -0.000959123, 0.00251636,
            0.00197985, -0.00819944, -0.00322287, 0.000494581, 0.00121647,
            -0.00819944, -0.00322287, 0.000494581, 0.00121647, -0.00682319,
            -0.000959123, 0.00251636, 0.00197985, -0.00535242, -0.00189859,
            0.000668578, 0.00114994, -0.00605613, -0.00360345, -0.00119241,
            5.07709e-05, -0.00605613, -0.00360345, -0.00119241, 5.07709e-05,
            -0.00535242, -0.00189859, 0.000668578, 0.00114994
        ])

        # some tensor data
        tensor1 = np.array([
            0.00486224, 0.0130364, 0.00474089, 0, 0, 0, -0.00486224,
            -0.0130364, -0.00474089, -0.000893302, 0.000415331, 0.00206496, 0,
            0, 0, 0.000893302, -0.000415331, -0.00206496, -0.000805675,
            -0.00339055, -0.000652415, 0, 0, 0, 0.000805675, 0.00339055,
            0.000652415, -0.000490711, -0.00294194, -0.00131963, 0, 0, 0,
            0.000490711, 0.00294194, 0.00131963
        ])

        tensor2 = np.array([
            0.0444173, 0.00136075, 5.56948e-07, 0.051794, 0.0153307,
            0.00474144, 0.0444173, 0.00136075, 5.56948e-07, 0.00163012,
            0.00446936, 0.000948565, 0.000269645, 0.00490565, 0.00301353,
            0.00163012, 0.00446936, 0.000948565, -0.00783519, 0.00186237,
            0.00148632, -0.00905312, -0.00176937, 0.000833901, -0.00783519,
            0.00186237, 0.00148632, -0.00715078, 0.000296698, 0.00129651,
            -0.00789121, -0.00285195, -2.31192e-05, -0.00715078, 0.000296698,
            0.00129651
        ])

        # divide data at different locations to make sure it does not get mixed
        tensors_element1 = np.array([tensor1, tensor2])
        tensors_element2 = tensors_element1 / 2.

        tensors_observation1 = np.array([tensors_element1, tensors_element2])
        tensors_observation2 = tensors_observation1 / 4.

        tensors_file1 = np.array([tensors_observation1, tensors_observation2])

        self.element_names = ["Element1", "Element2"]
        self.tensor_names = ["tensor1", "tensor2"]

        self.observation_ids = [42, 1000]
        self.observation_values = [1., 3.]

        target_vol = file1.get_vol(self.volume_name)
        for j, observation in enumerate(tensors_file1):
            volume_data = []
            for k, element in enumerate(observation):
                components = []
                for l, tensor in enumerate(element):
                    components.append(
                        TensorComponent(
                            os.path.join(self.element_names[k],
                                         self.tensor_names[l]),
                            DataVector(tensor, copy=False)))

                # this matches the solution by sol
                volume_data.append(
                    ElementVolumeData([3, 3, 4], components, [
                        Spectral.Basis.Legendre, Spectral.Basis.Legendre,
                        Spectral.Basis.Legendre
                    ], [
                        Spectral.Quadrature.GaussLobatto,
                        Spectral.Quadrature.GaussLobatto,
                        Spectral.Quadrature.GaussLobatto
                    ]))

            target_vol.write_volume_data(self.observation_ids[j],
                                         self.observation_values[j],
                                         volume_data)

    def tearDown(self):
        os.remove(self.file_name)
        try:
            os.remove(os.path.join(self.path, "interpolated_1.h5"))
        except OSError:
            pass

    def test_simple_interpolation(self):

        mesh = Spectral.Mesh3D([4, 4, 4], [Spectral.Basis.Legendre] * 3,
                               [Spectral.Quadrature.Gauss] * 3)

        target_volume_name = "/VolumeDataInterpolated"
        InterpolateVolumeData.interpolate_h5_file(self.file_name, mesh,
                                                  self.file_name,
                                                  self.volume_name,
                                                  target_volume_name)

        file = spectre_h5.H5File(self.file_name, "r")

        vol = file.get_vol(target_volume_name)

        self.assertEqual(vol.get_dimension(), 3)

        #python2 support...
        try:
            self.assertItemsEqual(vol.list_observation_ids(),
                                  self.observation_ids)
        except AttributeError:
            self.assertCountEqual(vol.list_observation_ids(),
                                  self.observation_ids)

        for i, obs in enumerate(self.observation_ids):
            self.assertEqual(vol.get_grid_names(obs), self.element_names)
            self.assertEqual(vol.get_observation_value(obs),
                             self.observation_values[i])
            self.assertEqual(vol.list_tensor_components(obs),
                             self.tensor_names)
            self.assertEqual(vol.get_extents(obs), [[4, 4, 4], [4, 4, 4]])
            self.assertEqual(vol.get_bases(obs), [["Legendre"] * 3] * 2)
            self.assertEqual(vol.get_quadratures(obs), [["Gauss"] * 3] * 2)

        tensor1_obs_1 = np.asarray(vol.get_tensor_component(42, 'tensor1'))
        tensor2_obs_1 = np.asarray(vol.get_tensor_component(42, 'tensor2'))

        tensor1_obs_2 = np.asarray(vol.get_tensor_component(1000, 'tensor1'))
        tensor2_obs_2 = np.asarray(vol.get_tensor_component(1000, 'tensor2'))

        file.close()

        self.assertTrue(np.allclose(tensor1_obs_1[:64], self.sol1, 1e-7, 1e-7))
        self.assertTrue(
            np.allclose(tensor1_obs_1[64:], self.sol1 / 2., 1e-7, 1e-7))
        self.assertTrue(np.allclose(tensor2_obs_1[:64], self.sol2, 1e-7, 1e-7))
        self.assertTrue(
            np.allclose(tensor2_obs_1[64:], self.sol2 / 2., 1e-7, 1e-7))

        self.assertTrue(
            np.allclose(tensor1_obs_2[:64], self.sol1 / 4., 1e-7, 1e-7))
        self.assertTrue(
            np.allclose(tensor1_obs_2[64:], self.sol1 / 8., 1e-7, 1e-7))
        self.assertTrue(
            np.allclose(tensor2_obs_2[:64], self.sol2 / 4., 1e-7, 1e-7))
        self.assertTrue(
            np.allclose(tensor2_obs_2[64:], self.sol2 / 8., 1e-7, 1e-7))


if __name__ == '__main__':
    unittest.main()
