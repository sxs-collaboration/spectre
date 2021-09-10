#Distributed under the MIT License.
#See LICENSE.txt for details.

from spectre.Spectral import Mesh1D, Mesh2D, Mesh3D
from spectre.Spectral import Basis, Quadrature
try:
    import cPickle as pickle  # Use cPickle on Python 2.7
except ImportError:
    import pickle

import numpy as np
import unittest
import random


class TestMesh(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        self.bases = [Basis.Legendre, Basis.Chebyshev, Basis.FiniteDifference]
        self.quadratures = [
            Quadrature.Gauss, Quadrature.GaussLobatto, Quadrature.CellCentered,
            Quadrature.FaceCentered
        ]
        self.extents = range(12)
        self.Mesh = [Mesh1D, Mesh2D, Mesh3D]

    def check_extents(self, mesh, extents):
        self.assertEqual(mesh.extents(0), extents[0])
        self.assertEqual(mesh.extents(), extents)
        self.assertEqual(mesh.number_of_grid_points(), np.prod(extents))

    def check_basis(self, mesh, bases):
        self.assertEqual(mesh.basis(0), bases[0])
        self.assertEqual(mesh.basis(), bases)

    def check_quadrature(self, mesh, quadratures):
        self.assertEqual(mesh.quadrature(0), quadratures[0])
        self.assertEqual(mesh.quadrature(), quadratures)

    def test_uniform(self):
        for dim in range(3):
            for extent in range(12):
                for basis in self.bases:
                    for quadrature in self.quadratures:
                        # the mesh constructor of dimension dim + 1
                        Mesh = self.Mesh[dim]
                        mesh = Mesh(extent, basis, quadrature)
                        self.assertEqual(mesh.dim, dim + 1)
                        self.check_extents(mesh,
                                           [extent for _ in range(dim + 1)])
                        self.check_basis(mesh, [basis for _ in range(dim + 1)])
                        self.check_quadrature(
                            mesh, [quadrature for _ in range(dim + 1)])

    def test_nonuniform_extents(self):
        for dim in range(3):
            for basis in self.bases:
                for quadrature in self.quadratures:
                    for i in range(100):
                        # the mesh constructor of dimension dim + 1
                        Mesh = self.Mesh[dim]
                        extents = [
                            random.choice(self.extents) for _ in range(dim + 1)
                        ]
                        mesh = Mesh(extents, basis, quadrature)
                        self.check_extents(mesh, extents)

    def test_nonuniform_all_with_pickle(self):
        for dim in range(3):
            for i in range(100):
                # the mesh constructor of dimension dim + 1
                Mesh = self.Mesh[dim]
                extents = [random.choice(self.extents) for _ in range(dim + 1)]
                bases = [random.choice(self.bases) for _ in range(dim + 1)]
                quadratures = [
                    random.choice(self.quadratures) for _ in range(dim + 1)
                ]

                mesh = Mesh(extents, bases, quadratures)
                # Use pickle protocol 2 for Py2 compatibility
                mesh = pickle.loads(pickle.dumps(mesh, protocol=2))
                self.check_extents(mesh, extents)
                self.check_basis(mesh, bases)
                self.check_quadrature(mesh, quadratures)

    def test_equality(self):
        for dim in range(3):
            for basis in self.bases:
                for quadrature in self.quadratures:
                    # the mesh constructor of dimension dim + 1
                    Mesh = self.Mesh[dim]
                    extents = [
                        random.choice(self.extents) for _ in range(dim + 1)
                    ]
                    mesh = Mesh(extents, basis, quadrature)
                    self.assertTrue(mesh == Mesh(extents, basis, quadrature))
                    self.assertFalse(mesh != Mesh(extents, basis, quadrature))
                    self.assertTrue(
                        mesh != Mesh([ex + 1
                                      for ex in extents], basis, quadrature))
                    self.assertFalse(
                        mesh == Mesh([ex + 1
                                      for ex in extents], basis, quadrature))


if __name__ == '__main__':
    unittest.main()
