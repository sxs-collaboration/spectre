# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import unittest

import numpy as np
import numpy.testing as npt

from spectre.Pipelines.EccentricityControl.InitialOrbitalParameters import (
    initial_orbital_parameters,
)
from spectre.support.Logging import configure_logging


class TestInitialOrbitalParameters(unittest.TestCase):
    def test_initial_orbital_parameters(self):
        np.set_printoptions(precision=14)
        npt.assert_allclose(
            initial_orbital_parameters(
                mass_ratio=1.0,
                dimensionless_spin_a=[0.0, 0.0, 0.0],
                dimensionless_spin_b=[0.0, 0.0, 0.0],
                separation=20.0,
                orbital_angular_velocity=0.01,
                radial_expansion_velocity=-1.0e-5,
            ),
            [20.0, 0.01, -1.0e-5],
        )
        npt.assert_allclose(
            initial_orbital_parameters(
                mass_ratio=1.0,
                dimensionless_spin_a=[0.0, 0.0, 0.0],
                dimensionless_spin_b=[0.0, 0.0, 0.0],
                eccentricity=0.0,
                separation=16.0,
            ),
            [16.0, 0.014454484323416913, -4.236562633362394e-05],
        )
        npt.assert_allclose(
            initial_orbital_parameters(
                mass_ratio=1.0,
                dimensionless_spin_a=[0.0, 0.0, 0.0],
                dimensionless_spin_b=[0.0, 0.0, 0.0],
                eccentricity=0.0,
                orbital_angular_velocity=0.015,
            ),
            [15.59033203125, 0.015, -4.696365029012517e-05],
        )
        npt.assert_allclose(
            initial_orbital_parameters(
                mass_ratio=1.0,
                dimensionless_spin_a=[0.0, 0.0, 0.0],
                dimensionless_spin_b=[0.0, 0.0, 0.0],
                eccentricity=0.0,
                orbital_angular_velocity=0.015,
            ),
            [15.59033203125, 0.015, -4.696365029012517e-05],
        )
        npt.assert_allclose(
            initial_orbital_parameters(
                mass_ratio=1.0,
                dimensionless_spin_a=[0.0, 0.0, 0.0],
                dimensionless_spin_b=[0.0, 0.0, 0.0],
                eccentricity=0.0,
                num_orbits=20,
            ),
            [15.71142578125, 0.014835205078125004, -4.554164727449197e-05],
        )
        npt.assert_allclose(
            initial_orbital_parameters(
                mass_ratio=1.0,
                dimensionless_spin_a=[0.0, 0.0, 0.0],
                dimensionless_spin_b=[0.0, 0.0, 0.0],
                eccentricity=0.0,
                time_to_merger=6000,
            ),
            [16.0909423828125, 0.01433787536621094, -4.14229775202535e-05],
        )


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
