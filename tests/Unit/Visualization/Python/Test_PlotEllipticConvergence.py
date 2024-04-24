# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import shutil
import unittest

import click
import numpy as np
from click.testing import CliRunner

import spectre.Informer as spectre_informer
import spectre.IO.H5 as spectre_h5
from spectre.Visualization.PlotEllipticConvergence import (
    plot_elliptic_convergence_command,
)


class TestPlotEllipticConvergence(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(
            spectre_informer.unit_test_build_path(),
            "Visualization/PlotEllipticConvergence",
        )
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.h5_filename = os.path.join(self.test_dir, "Residuals.h5")
        os.makedirs(self.test_dir, exist_ok=True)
        with spectre_h5.H5File(self.h5_filename, "w") as h5_file:
            linear_residuals = h5_file.insert_dat(
                "/GmresResiduals",
                legend=["Iteration", "Walltime", "Residual"],
                version=0,
            )
            linear_residuals.append([0, 0.1, 1.0])
            linear_residuals.append([1, 0.2, 0.5])
            linear_residuals.append([0, 0.3, 0.6])
            linear_residuals.append([1, 0.5, 0.4])
            linear_residuals.append([2, 0.8, 0.2])
            h5_file.close_current_object()
            nonlinear_residuals = h5_file.insert_dat(
                "/NewtonRaphsonResiduals",
                legend=["Iteration", "Walltime", "Residual"],
                version=0,
            )
            nonlinear_residuals.append([0, 0.1, 1.0])
            nonlinear_residuals.append([1, 0.3, 0.5])

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_cli(self):
        output_filename = os.path.join(self.test_dir, "output.pdf")
        runner = CliRunner()
        result = runner.invoke(
            plot_elliptic_convergence_command,
            [
                self.h5_filename,
                "-o",
                output_filename,
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)
        # We can't test that the plot is correct, so just make sure it at least
        # gets written out.
        self.assertTrue(os.path.exists(output_filename))


if __name__ == "__main__":
    unittest.main(verbosity=2)
