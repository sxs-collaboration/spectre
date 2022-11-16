# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.DataStructures import DataVector
from spectre.IO.H5 import TensorComponent, ElementVolumeData
from spectre.Spectral import Basis, Quadrature
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

    def test_element_volume_data(self):
        # Set up Extents and Tensor Volume data
        tensor_component_1 = TensorComponent("tensor component one",
                                             DataVector([1.5, 1.1]))
        tensor_component_2 = TensorComponent("tensor component two",
                                             DataVector([7.1, 5]))
        basis = Basis.Legendre
        quad = Quadrature.Gauss
        element_data = ElementVolumeData(
            element_name="grid_name",
            components=[tensor_component_1, tensor_component_2],
            extents=[3, 4],
            basis=2 * [basis],
            quadrature=2 * [quad])

        # Test extents
        self.assertEqual(element_data.extents, [3, 4])
        element_data.extents = [5, 6]
        self.assertEqual(element_data.extents, [5, 6])
        # Test tensor components
        self.assertEqual(element_data.tensor_components,
                         [tensor_component_1, tensor_component_2])
        element_data.tensor_components = [
            tensor_component_1, tensor_component_1
        ]
        self.assertEqual(element_data.tensor_components,
                         [tensor_component_1, tensor_component_1])
        # Test basis and quadrature
        self.assertEqual(element_data.basis, [basis, basis])
        self.assertEqual(element_data.quadrature, [quad, quad])
        self.assertEqual(element_data.element_name, "grid_name")


if __name__ == '__main__':
    unittest.main(verbosity=2)
