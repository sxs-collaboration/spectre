# Distributed under the MIT License.
# See LICENSE.txt for details.

import pickle
import random
import unittest

import numpy as np

from spectre.Spectral import Basis, Mesh, Quadrature


class TestMesh(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        self.bases = [Basis.Legendre, Basis.Chebyshev, Basis.FiniteDifference]
        self.quadratures = [
            Quadrature.Gauss,
            Quadrature.GaussLobatto,
            Quadrature.CellCentered,
            Quadrature.FaceCentered,
        ]
        self.extents = range(12)

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
        for dim in [1, 2, 3]:
            for extent in range(12):
                for basis in self.bases:
                    for quadrature in self.quadratures:
                        mesh = Mesh[dim](extent, basis, quadrature)
                        self.assertEqual(mesh.dim, dim)
                        self.check_extents(mesh, [extent for _ in range(dim)])
                        self.check_basis(mesh, [basis for _ in range(dim)])
                        self.check_quadrature(
                            mesh, [quadrature for _ in range(dim)]
                        )

    def test_nonuniform_extents(self):
        for dim in [1, 2, 3]:
            for basis in self.bases:
                for quadrature in self.quadratures:
                    for i in range(100):
                        extents = [
                            random.choice(self.extents) for _ in range(dim)
                        ]
                        mesh = Mesh[dim](extents, basis, quadrature)
                        self.check_extents(mesh, extents)

    def test_nonuniform_all_with_pickle(self):
        for dim in [1, 2, 3]:
            for i in range(100):
                extents = [random.choice(self.extents) for _ in range(dim)]
                bases = [random.choice(self.bases) for _ in range(dim)]
                quadratures = [
                    random.choice(self.quadratures) for _ in range(dim)
                ]

                mesh = Mesh[dim](extents, bases, quadratures)
                mesh = pickle.loads(pickle.dumps(mesh))
                self.check_extents(mesh, extents)
                self.check_basis(mesh, bases)
                self.check_quadrature(mesh, quadratures)

    def test_equality(self):
        for dim in [1, 2, 3]:
            for basis in self.bases:
                for quadrature in self.quadratures:
                    extents = [random.choice(self.extents) for _ in range(dim)]
                    mesh = Mesh[dim](extents, basis, quadrature)
                    self.assertTrue(
                        mesh == Mesh[dim](extents, basis, quadrature)
                    )
                    self.assertFalse(
                        mesh != Mesh[dim](extents, basis, quadrature)
                    )
                    self.assertTrue(
                        mesh
                        != Mesh[dim](
                            [ex + 1 for ex in extents], basis, quadrature
                        )
                    )
                    self.assertFalse(
                        mesh
                        == Mesh[dim](
                            [ex + 1 for ex in extents], basis, quadrature
                        )
                    )

    def test_slices(self):
        for dim in [1, 2, 3]:
            for basis in self.bases:
                for quadrature in self.quadratures:
                    extents = [random.choice(self.extents) for _ in range(dim)]
                    mesh = Mesh[dim](extents, basis, quadrature)
                    self.assertEqual(
                        mesh.slices(),
                        [
                            Mesh[1](extent, basis, quadrature)
                            for extent in extents
                        ],
                    )


if __name__ == "__main__":
    unittest.main()
