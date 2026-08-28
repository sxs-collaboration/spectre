# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

import numpy as np
import numpy.testing as npt

from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import Frame, Scalar, tnsr
from spectre.Domain import ElementId
from spectre.Domain.Creators import SphericalShells
from spectre.Evolution.Systems.ScalarWave import (
    sw_b3_power_monitors,
    sw_shell_power_monitors,
)
from spectre.Spectral import Basis, Mesh, Quadrature, logical_coordinates


class TestSwSphericalShellPowerMonitor(unittest.TestCase):
    def setUp(self):
        self.inner_radius = 4.0
        self.outer_radius = 7.0
        self.radial_points = 5
        self.l_max = 3
        self.mesh = Mesh[3](
            [self.radial_points, self.l_max + 1, 2 * self.l_max + 1],
            [
                Basis.Legendre,
                Basis.SphericalHarmonic,
                Basis.SphericalHarmonic,
            ],
            [
                Quadrature.GaussLobatto,
                Quadrature.Gauss,
                Quadrature.Equiangular,
            ],
        )
        self.domain = SphericalShells(
            self.inner_radius,
            self.outer_radius,
            0,
            self.radial_points,
            self.l_max,
        ).create_domain()
        self.element_id = ElementId[3](0)
        self.num_points = self.mesh.number_of_grid_points()

    def _zero_fields(self):
        psi = Scalar[DataVector](num_points=self.num_points, fill=0.0)
        pi = Scalar[DataVector](num_points=self.num_points, fill=0.0)
        phi = tnsr.i[DataVector, 3, Frame.Inertial](
            num_points=self.num_points, fill=0.0
        )
        return psi, pi, phi

    def _call(self, psi, pi, phi):
        return sw_shell_power_monitors(
            psi,
            pi,
            phi,
            self.mesh,
            self.element_id,
            self.domain,
            0.0,
            {},
        )

    def test_result_structure(self):
        """Result dict has the expected keys and array lengths."""
        psi, pi, phi = self._zero_fields()
        monitors = self._call(psi, pi, phi)

        self.assertEqual(set(monitors.keys()), {"Psi", "Pi", "Phi"})
        for variable in ("Psi", "Pi", "Phi"):
            self.assertEqual(
                set(monitors[variable].keys()), {"radial", "angular"}
            )
            self.assertEqual(
                len(monitors[variable]["radial"]), self.radial_points
            )
            self.assertEqual(len(monitors[variable]["angular"]), self.l_max + 1)

    def test_psi_isolation(self):
        """Setting only Psi gives zero Pi and Phi monitors."""
        psi, pi, phi = self._zero_fields()
        logical_coords = np.asarray(logical_coordinates(self.mesh))
        # Purely radial profile: uniform in angle (excites only l=0 SH mode).
        radial = DataVector(
            0.5 * (self.outer_radius - self.inner_radius) * logical_coords[0]
            + 0.5 * (self.outer_radius + self.inner_radius)
        )
        psi[0] = radial

        monitors = self._call(psi, pi, phi)

        self.assertGreater(max(monitors["Psi"]["radial"]), 0.0)
        self.assertGreater(monitors["Psi"]["angular"][0], 0.0)
        npt.assert_array_equal(np.asarray(monitors["Pi"]["radial"]), 0.0)
        npt.assert_array_equal(np.asarray(monitors["Pi"]["angular"]), 0.0)
        npt.assert_array_equal(np.asarray(monitors["Phi"]["radial"]), 0.0)
        npt.assert_array_equal(np.asarray(monitors["Phi"]["angular"]), 0.0)

    def test_pi_isolation(self):
        """Setting only Pi gives zero Psi and Phi monitors."""
        psi, pi, phi = self._zero_fields()
        logical_coords = np.asarray(logical_coordinates(self.mesh))
        radial = DataVector(
            0.5 * (self.outer_radius - self.inner_radius) * logical_coords[0]
            + 0.5 * (self.outer_radius + self.inner_radius)
        )
        pi[0] = radial

        monitors = self._call(psi, pi, phi)

        self.assertGreater(max(monitors["Pi"]["radial"]), 0.0)
        self.assertGreater(monitors["Pi"]["angular"][0], 0.0)
        npt.assert_array_equal(np.asarray(monitors["Psi"]["radial"]), 0.0)
        npt.assert_array_equal(np.asarray(monitors["Psi"]["angular"]), 0.0)
        npt.assert_array_equal(np.asarray(monitors["Phi"]["radial"]), 0.0)
        npt.assert_array_equal(np.asarray(monitors["Phi"]["angular"]), 0.0)

    def test_phi_isolation(self):
        """Setting only Phi_x gives zero Psi and Pi monitors."""
        psi, pi, phi = self._zero_fields()
        logical_coords = np.asarray(logical_coordinates(self.mesh))
        radial = DataVector(
            0.5 * (self.outer_radius - self.inner_radius) * logical_coords[0]
            + 0.5 * (self.outer_radius + self.inner_radius)
        )
        phi[phi.get_storage_index(0)] = radial

        monitors = self._call(psi, pi, phi)

        self.assertGreater(max(monitors["Phi"]["radial"]), 0.0)
        npt.assert_array_equal(np.asarray(monitors["Psi"]["radial"]), 0.0)
        npt.assert_array_equal(np.asarray(monitors["Psi"]["angular"]), 0.0)
        npt.assert_array_equal(np.asarray(monitors["Pi"]["radial"]), 0.0)
        npt.assert_array_equal(np.asarray(monitors["Pi"]["angular"]), 0.0)

    def test_psi_pi_equal_for_same_profile(self):
        """Psi and Pi monitors are identical when given the same profile."""
        logical_coords = np.asarray(logical_coordinates(self.mesh))
        profile = DataVector(
            0.5 * (self.outer_radius - self.inner_radius) * logical_coords[0]
            + 0.5 * (self.outer_radius + self.inner_radius)
        )

        psi, pi, phi = self._zero_fields()
        psi[0] = profile
        m_psi = self._call(psi, pi, phi)

        psi2, pi2, phi2 = self._zero_fields()
        pi2[0] = profile
        m_pi = self._call(psi2, pi2, phi2)

        npt.assert_allclose(
            np.asarray(m_psi["Psi"]["radial"]),
            np.asarray(m_pi["Pi"]["radial"]),
            rtol=1.0e-13,
        )
        npt.assert_allclose(
            np.asarray(m_psi["Psi"]["angular"]),
            np.asarray(m_pi["Pi"]["angular"]),
            rtol=1.0e-13,
        )

    def test_incompatible_mesh_raises(self):
        """A Legendre-only mesh raises RuntimeError."""
        psi, pi, phi = self._zero_fields()
        bad_mesh = Mesh[3](
            self.mesh.extents(),
            [Basis.Legendre, Basis.Legendre, Basis.Legendre],
            [
                Quadrature.GaussLobatto,
                Quadrature.GaussLobatto,
                Quadrature.GaussLobatto,
            ],
        )
        with self.assertRaisesRegex(
            RuntimeError, "require the mesh dimensions"
        ):
            sw_shell_power_monitors(
                psi,
                pi,
                phi,
                bad_mesh,
                self.element_id,
                self.domain,
                0.0,
                {},
            )

    def test_b3_result_structure(self):
        """sw_b3_power_monitors returns the expected dict structure."""
        n_r = 3
        l_max = 2
        b3_mesh = Mesh[3](
            [n_r, l_max + 1, 2 * l_max + 1],
            [Basis.ZernikeB3, Basis.ZernikeB3, Basis.ZernikeB3],
            [
                Quadrature.GaussRadauUpper,
                Quadrature.Gauss,
                Quadrature.Equiangular,
            ],
        )
        num_points = b3_mesh.number_of_grid_points()
        psi = Scalar[DataVector](num_points=num_points, fill=1.0)
        pi = Scalar[DataVector](num_points=num_points, fill=0.0)
        phi = tnsr.i[DataVector, 3, Frame.Inertial](
            num_points=num_points, fill=0.0
        )

        monitors = sw_b3_power_monitors(
            psi,
            pi,
            phi,
            b3_mesh,
            self.element_id,
            self.domain,
            0.0,
            {},
        )

        self.assertEqual(set(monitors.keys()), {"Psi", "Pi", "Phi"})
        for variable in ("Psi", "Pi", "Phi"):
            self.assertEqual(
                set(monitors[variable].keys()), {"radial", "angular"}
            )
            self.assertEqual(len(monitors[variable]["radial"]), n_r)
            self.assertEqual(len(monitors[variable]["angular"]), l_max + 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
