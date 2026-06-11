#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import shutil
import unittest

import h5py
import numpy as np
import numpy.testing as npt
from click.testing import CliRunner

import spectre.Informer as spectre_informer
import spectre.IO.H5 as spectre_h5
from spectre.IO.H5.CombineH5Dat import combine_h5_dat, combine_h5_dat_command


class TestCombineH5Dat(unittest.TestCase):
    def setUp(self):
        unit_test_build_dir = spectre_informer.unit_test_build_path()
        self.test_dir = os.path.join(unit_test_build_dir, "IO/H5/Python")
        os.makedirs(self.test_dir, exist_ok=True)

        # Set up directories to hold the input files and the joined output file
        self.input_dir = os.path.join(self.test_dir, "TestCombineH5DatInput")
        self.output_dir = os.path.join(self.test_dir, "TestCombineH5DatOutput")
        if os.path.exists(self.input_dir):
            shutil.rmtree(self.input_dir)
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        self.input_file_paths = [
            os.path.join(self.input_dir, "TestDatSeg" + str(i) + ".h5")
            for i in range(1, 6, 1)
        ]
        self.output_file_path = os.path.join(
            self.output_dir, "TestDatSegJoined.h5"
        )
        self.output_file_path_cli = os.path.join(
            self.output_dir, "TestDatSegJoinedCLI.h5"
        )
        self.expected_file_path = os.path.join(
            self.output_dir, "TestDatSegExpected.h5"
        )
        self.output_file_path_overlap = os.path.join(
            self.output_dir, "TestDatSegJoinedOverlap.h5"
        )
        self.expected_file_path_overlap = os.path.join(
            self.output_dir, "TestDatSegExpectedOverlap.h5"
        )

        # Generate sample dat data for H5 files to be joined
        self.wave_1 = np.array(
            [[t, np.sin(t), np.cos(t)] for t in np.arange(0, 10.0, 0.1)]
        )
        # Reverse order to test time sorting for the first h5 file
        self.reverse_wave_1 = self.wave_1[np.argsort(-self.wave_1[:, 0])]
        self.wave_2 = np.array(
            [[t, np.sin(t), np.cos(t)] for t in np.arange(10.0, 20.0, 0.1)]
        )
        self.wave_3 = np.array(
            [[t, np.sin(t), np.cos(t)] for t in np.arange(20.0, 30.0, 0.1)]
        )
        # Overlapping segment data to test remove_overlapping_segments option
        self.wave_4 = np.array(
            [[t, np.sin(t), np.cos(t)] for t in np.arange(29.0, 40.0, 0.1)]
        )
        # Reverse order to test time sorting within each segment/H5 file
        self.reverse_wave_4 = self.wave_4[np.argsort(-self.wave_4[:, 0])]
        self.wave_joined = np.concatenate(
            (self.wave_1, self.wave_2, self.wave_3), axis=0
        )
        # Joined data that preserves later times/data
        self.wave_joined_overlap = np.concatenate(
            (self.wave_1, self.wave_2, self.wave_3[:-10], self.wave_4), axis=0
        )
        self.pow_1 = np.array(
            [[t, t**2, t**3, t**4] for t in np.arange(0, 10.0, 0.1)]
        )
        self.pow_2 = np.array(
            [[t, t**2, t**3, t**4] for t in np.arange(10.0, 20.0, 0.1)]
        )
        self.pow_3 = np.array(
            [[t, t**2, t**3, t**4] for t in np.arange(20.0, 30.0, 0.1)]
        )
        self.pow_4 = np.array(
            [[t, t**2, t**3, t**4] for t in np.arange(29.0, 40.0, 0.1)]
        )
        self.pow_joined = np.concatenate(
            (self.pow_1, self.pow_2, self.pow_3), axis=0
        )
        self.pow_joined_overlap = np.concatenate(
            (self.pow_1, self.pow_2, self.pow_3[:-10], self.pow_4), axis=0
        )

        # Generate 4 H5 files with two dat files inside each
        with spectre_h5.H5File(
            file_name=self.input_file_paths[0], mode="r+"
        ) as h5file:
            wave_datfile = h5file.insert_dat(
                path="/Waves", legend=["Time", "Sin(t)", "Cos(t)"], version=0
            )
            wave_datfile.append(self.reverse_wave_1)
        with spectre_h5.H5File(
            file_name=self.input_file_paths[0], mode="r+"
        ) as h5file:
            pow_datfile = h5file.insert_dat(
                path="/Powers/Pow",
                legend=["Time", "t*t", "t*t*t", "t*t*t*t"],
                version=0,
            )
            pow_datfile.append(self.pow_1)
        with spectre_h5.H5File(
            file_name=self.input_file_paths[1], mode="r+"
        ) as h5file:
            wave_datfile = h5file.insert_dat(
                path="/Waves", legend=["Time", "Sin(t)", "Cos(t)"], version=0
            )
            wave_datfile.append(self.wave_1)
        with spectre_h5.H5File(
            file_name=self.input_file_paths[1], mode="r+"
        ) as h5file:
            pow_datfile = h5file.insert_dat(
                path="/Powers/Pow",
                legend=["Time", "t*t", "t*t*t", "t*t*t*t"],
                version=0,
            )
            pow_datfile.append(self.pow_1)
        with spectre_h5.H5File(
            file_name=self.input_file_paths[2], mode="r+"
        ) as h5file:
            wave_datfile = h5file.insert_dat(
                path="/Waves", legend=["Time", "Sin(t)", "Cos(t)"], version=0
            )
            wave_datfile.append(self.wave_2)
        with spectre_h5.H5File(
            file_name=self.input_file_paths[2], mode="r+"
        ) as h5file:
            pow_datfile = h5file.insert_dat(
                path="/Powers/Pow",
                legend=["Time", "t*t", "t*t*t", "t*t*t*t"],
                version=0,
            )
            pow_datfile.append(self.pow_2)
        with spectre_h5.H5File(
            file_name=self.input_file_paths[3], mode="r+"
        ) as h5file:
            wave_datfile = h5file.insert_dat(
                path="/Waves", legend=["Time", "Sin(t)", "Cos(t)"], version=0
            )
            wave_datfile.append(self.wave_3)
        with spectre_h5.H5File(
            file_name=self.input_file_paths[3], mode="r+"
        ) as h5file:
            pow_datfile = h5file.insert_dat(
                path="/Powers/Pow",
                legend=["Time", "t*t", "t*t*t", "t*t*t*t"],
                version=0,
            )
            pow_datfile.append(self.pow_3)
        with spectre_h5.H5File(
            file_name=self.input_file_paths[4], mode="r+"
        ) as h5file:
            wave_datfile = h5file.insert_dat(
                path="/Waves", legend=["Time", "Sin(t)", "Cos(t)"], version=0
            )
            wave_datfile.append(self.reverse_wave_4)
        with spectre_h5.H5File(
            file_name=self.input_file_paths[4], mode="r+"
        ) as h5file:
            pow_datfile = h5file.insert_dat(
                path="/Powers/Pow",
                legend=["Time", "t*t", "t*t*t", "t*t*t*t"],
                version=0,
            )
            pow_datfile.append(self.pow_4)

        self.test_yaml = """
        # Distributed under the MIT License.
        # See LICENSE.txt for details.
        Amplitude: 1.0 # Code units
        Frequency: 1.0 # Hz
        Phase: 0.0     # Rad
        Powers: [1, 2, 3, 4]
        """
        with h5py.File(self.input_file_paths[0], "r+") as h5file:
            h5file.attrs.modify("InputSource.yaml", self.test_yaml)
        with h5py.File(self.input_file_paths[1], "r+") as h5file:
            h5file.attrs.modify("InputSource.yaml", self.test_yaml)
        with h5py.File(self.input_file_paths[2], "r+") as h5file:
            h5file.attrs.modify("InputSource.yaml", self.test_yaml)
        with h5py.File(self.input_file_paths[3], "r+") as h5file:
            h5file.attrs.modify("InputSource.yaml", self.test_yaml)
        with h5py.File(self.input_file_paths[4], "r+") as h5file:
            h5file.attrs.modify("InputSource.yaml", self.test_yaml)

    def tearDown(self):
        if os.path.exists(self.input_dir):
            shutil.rmtree(self.input_dir)
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_combine_h5_dat(self):
        combine_h5_dat(
            output=self.output_file_path,
            h5files=self.input_file_paths[1:4],
            remove_overlapping_segments=False,
            force=None,
        )
        combine_h5_dat(
            output=self.output_file_path_overlap,
            h5files=self.input_file_paths,
            remove_overlapping_segments=True,
            force=None,
        )

        with h5py.File(self.output_file_path) as h5file:
            npt.assert_allclose(h5file["Waves.dat"], self.wave_joined)
            npt.assert_allclose(h5file["Powers/Pow.dat"], self.pow_joined)
            self.assertEqual(h5file.attrs["InputSource.yaml"], self.test_yaml)
        with h5py.File(self.output_file_path_overlap) as h5file:
            npt.assert_allclose(h5file["Waves.dat"], self.wave_joined_overlap)
            npt.assert_allclose(
                h5file["Powers/Pow.dat"], self.pow_joined_overlap
            )
            self.assertEqual(h5file.attrs["InputSource.yaml"], self.test_yaml)

    def test_cli(self):
        runner = CliRunner()
        result = runner.invoke(
            combine_h5_dat_command,
            ["-o", self.output_file_path_cli, *self.input_file_paths[:3]],
            catch_exceptions=False,
        )
        with self.assertRaisesRegex(
            ValueError, "exists; to overwrite, use --force"
        ):
            runner.invoke(
                combine_h5_dat_command,
                ["-o", self.output_file_path_cli, *self.input_file_paths[:3]],
                catch_exceptions=False,
            )
        result_force = runner.invoke(
            combine_h5_dat_command,
            [
                "-o",
                self.output_file_path_cli,
                "--force",
                *self.input_file_paths[1:4],
            ],
            catch_exceptions=False,
        )
        result_overlap = runner.invoke(
            combine_h5_dat_command,
            [
                "-o",
                self.output_file_path_cli,
                "--force",
                "--remove-overlapping-segments",
                *self.input_file_paths,
            ],
            catch_exceptions=False,
        )
        with h5py.File(self.output_file_path_cli) as h5file:
            npt.assert_allclose(h5file["Waves.dat"], self.wave_joined_overlap)
            npt.assert_allclose(
                h5file["Powers/Pow.dat"], self.pow_joined_overlap
            )
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result_force.exit_code, 0)
        self.assertEqual(result_overlap.exit_code, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
