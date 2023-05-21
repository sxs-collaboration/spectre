# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import unittest

import click
import h5py
import numpy as np
from click.testing import CliRunner

import spectre.Informer as spectre_informer
from spectre.Visualization.PlotDatFile import plot_dat_command


class TestPlotDatFile(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.join(
            spectre_informer.unit_test_src_path(), "Visualization/Python"
        )
        self.filename = os.path.join(self.data_dir, "DatTestData.h5")
        self.stylesheet_filename = os.path.join(
            self.data_dir, "teststyle.mplstyle"
        )
        self.test_dir = os.path.join(
            spectre_informer.unit_test_build_path(),
            "Visualization/Python/PlotDatFile",
        )
        self.runner = CliRunner()

    def test_list_subfiles(self):
        result = self.runner.invoke(
            plot_dat_command, [self.filename], catch_exceptions=False
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("TimeSteps2.dat", result.output)

    def test_nonexistent_subfile(self):
        result = self.runner.invoke(
            plot_dat_command,
            [self.filename, "-d", "TimeSteps"],
            catch_exceptions=False,
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Unable to open dat file 'TimeSteps.dat'.", result.output)

    def test_legend_only(self):
        # Invoke with the subfile name but no functions to plot
        result = self.runner.invoke(
            plot_dat_command,
            [self.filename, "-d", "TimeSteps2"],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Time", result.output)
        self.assertIn("NumberOfPoints", result.output)
        self.assertIn("Slab size", result.output)
        # Invoke with the "-l" option and a couple of functions
        result = self.runner.invoke(
            plot_dat_command,
            [self.filename, "-d", "TimeSteps2", "-l", "-y", "NumberOfPoints"],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Time", result.output)
        self.assertIn("NumberOfPoints", result.output)
        self.assertIn("Slab size", result.output)

    def test_write_plot(self):
        output_filename = os.path.join(self.test_dir, "output.pdf")
        os.makedirs(self.test_dir, exist_ok=True)

        # Run with minimal options
        result = self.runner.invoke(
            plot_dat_command,
            [
                self.filename,
                "-d",
                "TimeSteps2",
                "-y",
                "Slab size",
                "-o",
                output_filename,
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)
        # We can't test that the plot is correct, so just make sure it at least
        # gets written out.
        self.assertTrue(os.path.exists(output_filename))

        # Run with all options
        result = self.runner.invoke(
            plot_dat_command,
            [
                self.filename,
                "-d",
                "Group0/MemoryData",
                "-x",
                "Time",
                "-y",
                "Usage in MB",
                "-o",
                output_filename,
                "--x-label",
                "The time",
                "--y-label",
                "The usage",
                "--x-logscale",
                "--y-logscale",
                "--x-bounds",
                "0.1",
                "1",
                "--y-bounds",
                "0.1",
                "0.2",
                "-t",
                "Memory Data",
                "-s",
                self.stylesheet_filename,
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)

        os.remove(output_filename)


if __name__ == "__main__":
    unittest.main(verbosity=2)
