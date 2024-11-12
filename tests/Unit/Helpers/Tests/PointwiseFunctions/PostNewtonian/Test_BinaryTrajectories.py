# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

import numpy as np
import numpy.testing as npt

from spectre.testing.PostNewtonian import BinaryTrajectories


class TestBinaryTrajectories(unittest.TestCase):
    def test_binary_trajectories(self):
        # This class is tested in C++. Only check here that Pybindings work.
        binary_trajectories = BinaryTrajectories(initial_separation=16)
        self.assertEqual(binary_trajectories.separation(0.0), 16.0)
        self.assertEqual(binary_trajectories.orbital_frequency(0.0), 0.015625)
        self.assertEqual(binary_trajectories.angular_velocity(0.0), 0.015625)
        npt.assert_allclose(
            binary_trajectories.positions(0.0),
            ([8.0, 0.0, 0.0], [-8.0, 0.0, 0.0]),
        )
        # Test vectorized interface
        times = np.linspace(0.0, 10.0, 10)
        npt.assert_allclose(
            binary_trajectories.positions(times),
            np.transpose(
                [binary_trajectories.positions(t) for t in times],
                axes=(1, 2, 0),
            ),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
