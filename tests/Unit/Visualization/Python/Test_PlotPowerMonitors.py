# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import shutil
import unittest

from click.testing import CliRunner

from spectre.Domain.Creators import Cylinder
from spectre.Informer import unit_test_build_path, unit_test_src_path
from spectre.Visualization.PlotPowerMonitors import (
    find_block_or_group,
    plot_power_monitors_command,
)


class TestPlotPowerMonitors(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(
            unit_test_build_path(), "Visualization", "PlotPowerMonitors"
        )
        os.makedirs(self.test_dir, exist_ok=True)
        self.h5_filename = os.path.join(
            unit_test_src_path(), "Visualization/Python", "VolTestData0.h5"
        )
        self.plot_filename = os.path.join(self.test_dir, "plot.pdf")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_find_block_or_group(self):
        domain = Cylinder(
            inner_radius=1.0,
            outer_radius=3.0,
            lower_bound=0.0,
            upper_bound=2.0,
            is_periodic_in_z=False,
            initial_refinement=1,
            initial_number_of_grid_points=[3, 4, 5],
            use_equiangular_map=True,
        ).create_domain()
        self.assertEqual(
            find_block_or_group(0, ["BlockyBlock", "InnerCube"], domain), 1
        )
        self.assertEqual(
            find_block_or_group(1, ["BlockyBlock", "InnerCube"], domain), None
        )
        self.assertEqual(
            find_block_or_group(1, ["InnerCube", "Wedges"], domain), 1
        )

    def test_cli(self):
        runner = CliRunner()
        # Test plotting a single step
        result = runner.invoke(
            plot_power_monitors_command,
            [
                self.h5_filename,
                "-d",
                "element_data",
                "--step",
                "-1",
                "-b",
                "Brick",
                "-e",
                "B*",
                "-y",
                "Psi",
                "-o",
                self.plot_filename,
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        # Can't easily test the plot itself, so just check that it was created
        self.assertTrue(os.path.exists(self.plot_filename))
        os.remove(self.plot_filename)

        # Test plotting over time
        result = runner.invoke(
            plot_power_monitors_command,
            [
                self.h5_filename,
                "-d",
                "element_data",
                "-b",
                "Brick",
                "-e",
                "B*",
                "-y",
                "Psi",
                "-o",
                self.plot_filename,
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        # Can't easily test the plot itself, so just check that it was created
        self.assertTrue(os.path.exists(self.plot_filename))
        os.remove(self.plot_filename)


if __name__ == "__main__":
    unittest.main(verbosity=2)
