# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import unittest

import numpy.testing as npt

from spectre.Pipelines.EccentricityControl.InitialOrbitalParameters import (
    initial_orbital_parameters,
)
from spectre.support.Logging import configure_logging


class TestInitialOrbitalParameters(unittest.TestCase):
    def test_initial_orbital_parameters(self):
        # Expected results are computed from SpEC's ZeroEccParamsFromPN.py
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
            [16.0, 0.014474280975952748, -4.117670632867514e-05],
        )
        npt.assert_allclose(
            initial_orbital_parameters(
                mass_ratio=1.0,
                dimensionless_spin_a=[0.0, 0.0, 0.0],
                dimensionless_spin_b=[0.0, 0.0, 0.0],
                eccentricity=0.0,
                orbital_angular_velocity=0.015,
            ),
            [15.6060791015625, 0.015, -4.541705362753467e-05],
        )
        npt.assert_allclose(
            initial_orbital_parameters(
                mass_ratio=1.0,
                dimensionless_spin_a=[0.0, 0.0, 0.0],
                dimensionless_spin_b=[0.0, 0.0, 0.0],
                eccentricity=0.0,
                orbital_angular_velocity=0.015,
            ),
            [15.6060791015625, 0.015, -4.541705362753467e-05],
        )
        npt.assert_allclose(
            initial_orbital_parameters(
                mass_ratio=1.0,
                dimensionless_spin_a=[0.0, 0.0, 0.0],
                dimensionless_spin_b=[0.0, 0.0, 0.0],
                eccentricity=0.0,
                num_orbits=20,
            ),
            [16.0421142578125, 0.014419921875000002, -4.0753460821644916e-05],
        )
        npt.assert_allclose(
            initial_orbital_parameters(
                mass_ratio=1.0,
                dimensionless_spin_a=[0.0, 0.0, 0.0],
                dimensionless_spin_b=[0.0, 0.0, 0.0],
                eccentricity=0.0,
                time_to_merger=6000,
            ),
            [16.1357421875, 0.01430025219917298, -3.9831982447244026e-05],
        )


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
