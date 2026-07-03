# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

import numpy as np
import numpy.testing as npt

from spectre.SpinWeightedSphericalHarmonics import (
    eth,
    ethbar,
    ethbar_eth,
    goldberg_to_nodal,
    nodal_to_goldberg,
)


def to_interleaved(complex_array):
    """Pack a complex array into the real [re, im, re, im, ...] layout the
    bindings consume."""
    interleaved = np.zeros(2 * len(complex_array))
    interleaved[0::2] = np.real(complex_array)
    interleaved[1::2] = np.imag(complex_array)
    return interleaved


def from_interleaved(interleaved):
    """Inverse of `to_interleaved`."""
    return np.asarray(interleaved)[0::2] + 1j * np.asarray(interleaved)[1::2]


class TestSwsh(unittest.TestCase):
    def setUp(self):
        self.l_max = 8
        self.rng = np.random.default_rng(20260626)

    def random_goldberg_modes(self):
        # Goldberg coefficients are indexed by (l, m) as l**2 + l + m, for
        # l = 0..l_max, so there are (l_max + 1)**2 complex modes.
        num_modes = (self.l_max + 1) ** 2
        return self.rng.uniform(-1.0, 1.0, num_modes) + 1j * self.rng.uniform(
            -1.0, 1.0, num_modes
        )

    def test_goldberg_nodal_round_trip(self):
        # nodal_to_goldberg is the inverse of goldberg_to_nodal, so a round trip
        # of band-limited spin-0 modes recovers the original coefficients.
        modes = self.random_goldberg_modes()
        nodal = goldberg_to_nodal(to_interleaved(modes), self.l_max, 0)
        recovered = from_interleaved(nodal_to_goldberg(nodal, self.l_max, 0))
        npt.assert_allclose(recovered, modes, rtol=1e-12, atol=1e-12)

    def test_ethbar_eth_matches_composition(self):
        # The composite operator ethbar_eth must equal applying eth and then
        # ethbar. This checks the individual eth and ethbar bindings against the
        # composite, independent of any normalization convention.
        modes = self.random_goldberg_modes()
        nodal = goldberg_to_nodal(to_interleaved(modes), self.l_max, 0)
        # eth raises spin 0 -> 1, ethbar lowers spin 1 -> 0
        eth_nodal = eth(nodal, self.l_max, 1, 0)
        composed = ethbar(eth_nodal, self.l_max, 1, 1)
        direct = ethbar_eth(nodal, self.l_max, 1, 0)
        npt.assert_allclose(
            from_interleaved(composed),
            from_interleaved(direct),
            rtol=1e-11,
            atol=1e-11,
        )

    def test_ethbar_eth_eigenvalue(self):
        # ethbar_eth is the spin-0 angular Laplacian: it multiplies the (l, m)
        # Goldberg coefficient by -l(l+1). Because it preserves spin weight, the
        # eigenvalue is identical in the Goldberg and libsharp representations.
        modes = self.random_goldberg_modes()
        nodal = goldberg_to_nodal(to_interleaved(modes), self.l_max, 0)
        result_modes = from_interleaved(
            nodal_to_goldberg(
                ethbar_eth(nodal, self.l_max, 1, 0), self.l_max, 0
            )
        )
        mode_index = np.arange((self.l_max + 1) ** 2)
        ell = np.floor(np.sqrt(mode_index)).astype(int)
        expected = -ell * (ell + 1) * modes
        npt.assert_allclose(result_modes, expected, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
