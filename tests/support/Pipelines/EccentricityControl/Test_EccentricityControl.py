# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import os
import shutil
import unittest

import numpy as np
import numpy.testing as npt
from click.testing import CliRunner

import spectre.IO.H5 as spectre_h5
from spectre.Informer import unit_test_build_path
from spectre.Pipelines.EccentricityControl import eccentricity_control_command
from spectre.Pipelines.EccentricityControl.EccentricityControl import (
    coordinate_separation_eccentricity_control,
)
from spectre.support.Logging import configure_logging


class TestEccentricityControl(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(
            unit_test_build_path(), "Pipelines", "EccentricityControl"
        )
        self.h5_filename = os.path.join(
            self.test_dir, "TestPlotTrajectoriesReductions.h5"
        )
        shutil.rmtree(self.test_dir, ignore_errors=True)
        os.makedirs(self.test_dir, exist_ok=True)
        self.create_h5_file()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def create_h5_file(self):
        """Create the H5"""
        logging.info(f"Creating HDF5 file: {self.h5_filename}")

        # Generate mock-up inspiral data
        nsamples = 120
        dt = 2.0

        # Mock eccentric orbit parameters
        ecc = 1e-2
        a = 16.0
        initial_phase = 0.0

        # Some useful definitions
        Omega_bar = np.sqrt(1.0 / (a**3))
        r_bar = a * (1 - ecc**2)

        # Origin offset and linear drift in z
        x0 = 0.35
        y0 = 0.35
        z0 = 0
        z1 = -9.0e-6

        # Save expected values for eccentricity oscillations
        self.initial_separation = r_bar
        self.initial_phase = initial_phase
        self.ecc = ecc
        self.frequency = Omega_bar
        self.amplitude = r_bar * ecc * Omega_bar
        logging.info(f"Expected eccentric parameters")
        logging.info(f"Frequency: {self.frequency}")
        logging.info(f"Amplitude of oscillations: {self.amplitude}")

        # Linear in eccentricity orbit approximation
        def r(t):
            return r_bar * (1 - ecc * np.cos(Omega_bar * t))

        def phase(t):
            return (
                Omega_bar * t
                + (2.0 * ecc / Omega_bar) * np.sin(Omega_bar * t)
                + initial_phase
            )

        # Define the spirals for equal mass orbits
        def SpiralA(t):
            return np.array(
                [
                    x0 + 0.5 * r(t) * np.cos(phase(t)),
                    y0 + 0.5 * r(t) * np.sin(phase(t)),
                    z0 + z1 * (1 - 0.1) * t,
                ]
            )

        def SpiralB(t):
            return np.array(
                [
                    x0 - 0.5 * r(t) * np.cos(phase(t)),
                    y0 - 0.5 * r(t) * np.sin(phase(t)),
                    z0 + z1 * (1 - 0.1) * t,
                ]
            )

        # Generate time samples
        tTable = np.arange(0, (nsamples + 1) * dt, dt)

        # Map time to spiral points
        AhA_data = np.array([[t, *SpiralA(t), *SpiralA(t)] for t in tTable])
        AhB_data = np.array([[t, *SpiralB(t), *SpiralB(t)] for t in tTable])

        with spectre_h5.H5File(self.h5_filename, "w") as h5_file:
            # Insert dataset for AhA
            dataset_AhA = h5_file.insert_dat(
                "ApparentHorizons/ControlSystemAhA_Centers.dat",
                legend=[
                    "Time",
                    "GridCenter_x",
                    "GridCenter_y",
                    "GridCenter_z",
                    "InertialCenter_x",
                    "InertialCenter_y",
                    "InertialCenter_z",
                ],  # Legend for the dataset
                version=0,  # Version number
            )
            # Populate dataset with AhA data
            for data_point in AhA_data:
                dataset_AhA.append(data_point)
            # Close dataset for AhA
            h5_file.close_current_object()

            # Insert dataset for AhB
            dataset_AhB = h5_file.insert_dat(
                "ApparentHorizons/ControlSystemAhB_Centers.dat",
                legend=[
                    "Time",
                    "GridCenter_x",
                    "GridCenter_y",
                    "GridCenter_z",
                    "InertialCenter_x",
                    "InertialCenter_y",
                    "InertialCenter_z",
                ],  # Legend for the dataset
                version=0,  # Version number
            )
            # Populate dataset with AhB data
            for data_point in AhB_data:
                dataset_AhB.append(data_point)
            # Close dataset for AhB
            h5_file.close_current_object()
        logging.info(
            f"Successfully created and populated HDF5 file: {self.h5_filename}"
        )

    def test_cli(self):
        output_filename = os.path.join(self.test_dir, "output.pdf")
        runner = CliRunner()
        result = runner.invoke(
            eccentricity_control_command,
            [
                self.h5_filename,
                "-A",
                "ApparentHorizons/ControlSystemAhA_Centers.dat",
                "-B",
                "ApparentHorizons/ControlSystemAhB_Centers.dat",
                "--tmin",
                0.0,
                "--tmax",
                500.0,
                "--angular-velocity-from-xcts",
                0.0173,
                "--expansion-from-xcts",
                -1e-6,
                "-o",
                output_filename,
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(os.path.exists(output_filename))

    def test_output_parameters(self):
        mock_angular_velocity_from_xcts = 0.0156
        mock_expansion_from_xcts = -1e-6

        functions = coordinate_separation_eccentricity_control(
            h5_file=self.h5_filename,
            subfile_name_aha="ApparentHorizons/ControlSystemAhA_Centers.dat",
            subfile_name_ahb="ApparentHorizons/ControlSystemAhB_Centers.dat",
            tmin=0.0,
            tmax=1200.0,
            angular_velocity_from_xcts=mock_angular_velocity_from_xcts,
            expansion_from_xcts=mock_expansion_from_xcts,
            fig=None,
        )

        for name, func in functions.items():
            logging.info(f"Test fit function: {name}")
            # Values from fit
            ampl, freq, phase = func["fit result"]["parameters"][:3]
            updates_from_fit = func["fit result"]["xcts updates"]
            updated_xcts_values_from_fit = func["fit result"][
                "updated xcts values"
            ]
            eccentricity_from_fit = func["fit result"]["eccentricity"]

            # Expected orbital values
            expected_magnitude_of_frequency = self.frequency
            expected_magnitude_of_oscillations = self.amplitude
            expected_magnitude_of_phase = np.pi / 2

            tol = 1e-6
            self.assertAlmostEqual(
                np.abs(ampl), expected_magnitude_of_oscillations, delta=tol
            )
            self.assertAlmostEqual(
                np.abs(freq), expected_magnitude_of_frequency, delta=tol
            )
            self.assertAlmostEqual(
                np.abs(phase), expected_magnitude_of_phase, delta=tol
            )

            # Expected updates
            expected_dOmg = (
                self.amplitude
                / 2.0
                / self.initial_separation
                * np.sin(self.initial_phase)
            )
            expected_dadot = (
                -self.amplitude
                / self.initial_separation
                * np.cos(self.initial_phase)
            )

            # Expected updates
            expected_updated_Omg = (
                mock_angular_velocity_from_xcts + expected_dOmg
            )
            expected_updated_adot = mock_expansion_from_xcts + expected_dadot

            # Updates from fit
            dOmg = updates_from_fit["omega update"]
            dadot = updates_from_fit["expansion update"]
            updated_Omg = updated_xcts_values_from_fit["omega"]
            updated_adot = updated_xcts_values_from_fit["expansion"]

            tol_for_updates = 5e-4
            self.assertAlmostEqual(expected_dOmg, dOmg, delta=tol_for_updates)
            self.assertAlmostEqual(expected_dadot, dadot, delta=tol_for_updates)
            self.assertAlmostEqual(
                expected_updated_Omg, updated_Omg, delta=tol_for_updates
            )
            self.assertAlmostEqual(
                expected_updated_adot, updated_adot, delta=tol_for_updates
            )

            tol_for_ecc = 5e-4
            self.assertAlmostEqual(
                self.ecc, eccentricity_from_fit, delta=tol_for_ecc
            )


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
