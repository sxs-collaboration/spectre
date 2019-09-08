# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.DataStructures import (DataVector, ExtentsAndTensorVolumeData,
                                    TensorComponent, Legendre, Gauss,
                                    ElementVolumeData)

import unittest
import numpy as np
import numpy.testing as npt


class TestTensorData(unittest.TestCase):
    # Tests for TensorComponent functions
    def test_tensor_component(self):
        # Set up Tensor Component
        tensor_component = TensorComponent("tensor component",
                                           DataVector([1.5, 1.1]))
        # Test name
        self.assertEqual(tensor_component.name, "tensor component")
        tensor_component.name = "new tensor component"
        self.assertEqual(tensor_component.name, "new tensor component")
        # Test data
        npt.assert_array_almost_equal(np.array(tensor_component.data),
                                      np.array([1.5, 1.1]))
        tensor_component.data = DataVector([6.7, 3.2])
        npt.assert_array_almost_equal(np.array(tensor_component.data),
                                      np.array([6.7, 3.2]))
        # Test str, repr
        self.assertEqual(str(tensor_component),
                         "(new tensor component, (6.7,3.2))")
        self.assertEqual(repr(tensor_component),
                         "(new tensor component, (6.7,3.2))")

    # Tests for ExtentsAndTensorVolumeData functions
    def test_extents_and_tensor_volume_data(self):
        # Set up Extents and Tensor Volume data
        tensor_component_1 = TensorComponent("tensor component one",
                                             DataVector([1.5, 1.1]))
        tensor_component_2 = TensorComponent("tensor component two",
                                             DataVector([7.1, 5]))
        extents_and_data = ExtentsAndTensorVolumeData(
            [1, 2, 3, 4], [tensor_component_1, tensor_component_2])
        # Test extents
        self.assertEqual(extents_and_data.extents, [1, 2, 3, 4])
        extents_and_data.extents = [5, 6, 7, 8]
        self.assertEqual(extents_and_data.extents, [5, 6, 7, 8])
        # Test tensor components
        self.assertEqual(extents_and_data.tensor_components,
                         [tensor_component_1, tensor_component_2])
        extents_and_data.tensor_components = [
            tensor_component_1, tensor_component_1
        ]
        self.assertEqual(extents_and_data.tensor_components,
                         [tensor_component_1, tensor_component_1])
        # Test str, repr
        self.assertEqual(
            str(extents_and_data),
            "((5,6,7,8),((tensor component one, (1.5,1.1)"
            "),(tensor component one, (1.5,1.1))))")
        self.assertEqual(
            repr(extents_and_data),
            "((5,6,7,8),((tensor component one, (1.5,1.1)"
            "),(tensor component one, (1.5,1.1))))")

        # Tests for ExtentsAndTensorVolumeData functions
    def test_element_volume_data(self):
        # Set up Extents and Tensor Volume data
        tensor_component_1 = TensorComponent("tensor component one",
                                             DataVector([1.5, 1.1]))
        tensor_component_2 = TensorComponent("tensor component two",
                                             DataVector([7.1, 5]))
        basis = Legendre
        quad = Gauss
        element_data = ElementVolumeData(
            [1, 2, 3, 4], [tensor_component_1, tensor_component_2],
            [basis, basis], [quad, quad])

        # Test extents
        self.assertEqual(element_data.extents, [1, 2, 3, 4])
        element_data.extents = [5, 6, 7, 8]
        self.assertEqual(element_data.extents, [5, 6, 7, 8])
        # Test tensor components
        self.assertEqual(element_data.tensor_components,
                         [tensor_component_1, tensor_component_2])
        element_data.tensor_components = [
            tensor_component_1, tensor_component_1
        ]
        self.assertEqual(element_data.tensor_components,
                         [tensor_component_1, tensor_component_1])
        # Test str, repr
        self.assertEqual(
            str(element_data), "((5,6,7,8),((tensor component one, (1.5,1.1)"
            "),(tensor component one, (1.5,1.1))))")
        self.assertEqual(
            repr(element_data), "((5,6,7,8),((tensor component one, (1.5,1.1)"
            "),(tensor component one, (1.5,1.1))))")
        # Test basis and quadrature
        self.assertEqual(element_data.basis, [basis, basis])
        self.assertEqual(element_data.quadrature, [quad, quad])


if __name__ == '__main__':
    unittest.main(verbosity=2)
