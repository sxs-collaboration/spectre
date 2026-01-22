# Distributed under the MIT License.
# See LICENSE.txt for details.


import random
import unittest

import numpy as np
import numpy.testing as npt

try:
    from scipy.special import sph_harm_y
except ImportError:
    # SciPy < 1.15
    from scipy.special import sph_harm

    def sph_harm_y(n, m, theta, phi):
        return sph_harm(m, n, phi, theta)


from spectre.SphericalHarmonics import Spherepack, SpherepackIterator


def convert_coef_to_spherepack(complex_coef, l, m):
    """
    converts spherical harmonic coefficients to spherepack coefficients,
    see https://spectre-code.org/classylm_1_1Spherepack.html
    """
    sign = 1.0 if m % 2 == 0 else -1.0
    part = np.real(complex_coef) if m >= 0 else np.imag(complex_coef)
    return sign * np.sqrt(2.0 / np.pi) * part


class TestStrahlkorper(unittest.TestCase):
    def test_sizes(self):
        for l in range(2, 6):
            for m in range(2, l + 1):
                spherepack = Spherepack(l, m)
                self.assertEqual(
                    spherepack.physical_size, (l + 1) * (2 * m + 1)
                )
                self.assertEqual(
                    spherepack.spectral_size, 2 * (l + 1) * (m + 1)
                )

    def test_spec_to_phys_and_back(self):
        l_max = random.randint(2, 12)
        m_max = random.randint(2, l_max)
        spherepack = Spherepack(l_max, m_max)
        thetas, phis = spherepack.theta_phi_points
        iterator = SpherepackIterator(l_max, m_max)
        spherepack_coefs = np.zeros(spherepack.spectral_size)
        scipy_collocation_values = np.zeros(
            np.shape(thetas), dtype=np.complex128
        )

        for l in range(0, l_max + 1):
            for m in range(0, min(l + 1, m_max + 1)):
                coef = random.uniform(-1.0, 1.0)
                if m > 0:
                    coef += random.uniform(-1.0, 1.0) * 1j
                    iterator.set(l, -m)
                    spherepack_coefs[iterator()] = convert_coef_to_spherepack(
                        coef, l, -m
                    )
                    # we are checking for a real field, so for the
                    # conjugate: a*(l,m) = (-1)^m a(l,m)
                    sign = 1.0 if m % 2 == 0 else -1.0
                    scipy_collocation_values += (
                        sign
                        * coef.conjugate()
                        * sph_harm_y(l, -m, thetas, phis)
                    )

                iterator.set(l, m)
                spherepack_coefs[iterator()] = convert_coef_to_spherepack(
                    coef, l, m
                )
                scipy_collocation_values += coef * sph_harm_y(
                    l, m, thetas, phis
                )

        npt.assert_array_almost_equal(
            np.zeros_like(thetas), np.imag(scipy_collocation_values)
        )
        collocation_values = np.asarray(
            spherepack.spec_to_phys(spherepack_coefs)
        )
        npt.assert_array_almost_equal(
            collocation_values, np.real(scipy_collocation_values)
        )

        recovered_coefs = np.asarray(
            spherepack.phys_to_spec(collocation_values)
        )
        npt.assert_array_almost_equal(
            recovered_coefs, np.asarray(spherepack_coefs)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
