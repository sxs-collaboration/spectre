# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

import numpy as np
import numpy.testing as npt

from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import (
    Frame,
    InverseJacobian,
    Jacobian,
    Scalar,
    tnsr,
)
from spectre.Domain import jacobian_diagnostic
from spectre.PointwiseFunctions.Punctures import adm_mass_integrand
from spectre.Spectral import Basis, Mesh, Quadrature


class TestTensor(unittest.TestCase):
    def test_tensor(self):
        coords = tnsr.I[DataVector, 3, Frame.Inertial](num_points=4, fill=0.0)
        spacetime_coords = tnsr.A[DataVector, 3, Frame.Inertial](
            num_points=1, fill=0.0
        )
        self.assertEqual(coords.rank, 1)
        self.assertEqual(coords.size, 3)
        self.assertEqual(coords.dim, 3)
        self.assertEqual(len(coords), 3)
        self.assertEqual(spacetime_coords.rank, coords.rank)
        self.assertEqual(spacetime_coords.size, coords.size + 1)
        self.assertEqual(spacetime_coords.dim, coords.dim + 1)
        self.assertEqual(len(spacetime_coords), len(coords) + 1)
        npt.assert_equal(coords[0], np.zeros(4))
        npt.assert_equal(coords[1], np.zeros(4))
        npt.assert_equal(coords[2], np.zeros(4))
        coords[0] = DataVector(4, 1.0)
        coords[1] = DataVector(4, 2.0)
        coords[2] = DataVector(4, 3.0)
        for d, xyz in enumerate(coords):
            npt.assert_equal(xyz, np.ones(4) * (d + 1))
            npt.assert_equal(xyz, coords.get(d))
            self.assertEqual(coords.multiplicity(d), 1)
            self.assertEqual(coords.component_suffix(d), ["_x", "_y", "_z"][d])

    def test_construct_from_list(self):
        data = [DataVector(xyz) for xyz in np.random.rand(3, 4)]
        coords = tnsr.I[DataVector, 3, Frame.Inertial](data)
        npt.assert_equal(np.array(coords), data)

    def test_numpy_interoperability(self):
        data = np.random.rand(3, 4)
        data_scalar = np.random.rand(4)
        data_symm_tensor = np.random.rand(6, 4)
        for copy in [True, False]:
            coords = tnsr.I[DataVector, 3, Frame.Inertial](data, copy=copy)
            for i, (a, b) in enumerate(zip(coords, data)):
                npt.assert_equal(a, b, f"Mismatch at index {i}")
            npt.assert_equal(np.array(coords), data)
            symm_tensor = tnsr.ii[DataVector, 3](data_symm_tensor, copy=copy)
            for i, (a, b) in enumerate(zip(symm_tensor, data_symm_tensor)):
                npt.assert_equal(a, b, f"Mismatch at index {i}")
            npt.assert_equal(np.array(symm_tensor), data_symm_tensor)
            # Construction of Scalar from 1D array
            scalar = Scalar[DataVector](data_scalar, copy=copy)
            npt.assert_equal(np.array(scalar), [data_scalar])
        with self.assertRaisesRegex(RuntimeError, "expected to be 2D"):
            tnsr.ii[DataVector, 3](np.random.rand(3, 3, 4))
        with self.assertRaisesRegex(RuntimeError, "3 independent components"):
            tnsr.I[DataVector, 3](np.random.rand(2, 4))
        # Implicit conversion from Numpy array to scalar
        adm_mass_integrand(
            field=np.random.rand(4),
            alpha=np.random.rand(4),
            beta=np.random.rand(4),
        )
        # Implicit conversion from Numpy array to vector
        mesh = Mesh[3](3, Basis.Legendre, Quadrature.GaussLobatto)
        jac = Jacobian[DataVector, 3](np.random.rand(9, 27))
        jacobian_diagnostic(
            jacobian=jac, inertial_coords=np.random.rand(3, 27), mesh=mesh
        )
        # Higher-rank tensor don't convert implicitly
        with self.assertRaisesRegex(
            TypeError, "incompatible function arguments"
        ):
            jacobian_diagnostic(
                jacobian=np.random.rand(9, 27),
                inertial_coords=np.random.rand(3, 27),
                mesh=Mesh[3](3, Basis.Legendre, Quadrature.GaussLobatto),
            )

    def test_buffer_strides(self):
        # The transpose should set up data with non-unit strides
        original_data = np.random.rand(4, 3)
        data = original_data.T
        coords = tnsr.I[DataVector, 3, Frame.Inertial](data)
        for i, (a, b) in enumerate(zip(coords, data)):
            npt.assert_equal(a, b, f"Mismatch at index {i}")
        npt.assert_equal(np.array(coords), data)
        # Non-owning DataVectors don't work with strides != 1
        with self.assertRaisesRegex(RuntimeError, "Non-owning"):
            coords = tnsr.I[DataVector, 3, Frame.Inertial](data, copy=False)

    def test_tensor_double(self):
        coords = tnsr.I[float, 3, Frame.Inertial](fill=0.0)
        coords[0] = 1.0
        coords[1] = 2.0
        coords[2] = 3.0
        npt.assert_equal(np.array(coords), [1.0, 2.0, 3.0])

    def test_scalar(self):
        scalar = Scalar[DataVector](num_points=4, fill=1.0)
        self.assertEqual(scalar.size, 1)
        self.assertEqual(scalar.rank, 0)
        self.assertEqual(scalar.dim, None)
        npt.assert_equal(np.array(scalar), np.ones((1, 4)))
        npt.assert_equal(scalar[0], scalar.get())

    def test_jacobian(self):
        jac = Jacobian[DataVector, 3](num_points=4, fill=1.0)
        inv_jac = InverseJacobian[DataVector, 3](num_points=4, fill=1.0)
        npt.assert_equal(np.array(jac), np.ones((9, 4)))
        npt.assert_equal(np.array(inv_jac), np.ones((9, 4)))
        self.assertEqual(jac.rank, 2)
        self.assertEqual(inv_jac.rank, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
