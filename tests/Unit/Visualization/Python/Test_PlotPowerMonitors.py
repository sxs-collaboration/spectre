# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import shutil
import unittest

import numpy as np
from click.testing import CliRunner

import spectre.IO.H5 as spectre_h5
from spectre.Domain import ElementId, serialize_domain
from spectre.Domain.Creators import Cylinder, SphericalShells
from spectre.Informer import unit_test_build_path, unit_test_src_path
from spectre.IO.H5 import ElementVolumeData, TensorComponent
from spectre.Spectral import Basis, Mesh, Quadrature, logical_coordinates
from spectre.Visualization.PlotPowerMonitors import (
    find_block_or_group,
    gh_sh_tensor_component_names,
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

        radial_points = 3
        l_max = 3
        shell_creator = SphericalShells(1.0, 2.0, 0, radial_points, l_max)
        shell_domain = shell_creator.create_domain()
        self.shell_block_name = shell_domain.blocks[0].name
        shell_mesh = Mesh[3](
            [radial_points, l_max + 1, 2 * l_max + 1],
            [
                Basis.Legendre,
                Basis.SphericalHarmonic,
                Basis.SphericalHarmonic,
            ],
            [
                Quadrature.GaussLobatto,
                Quadrature.Gauss,
                Quadrature.Equiangular,
            ],
        )
        logical_coords = np.asarray(logical_coordinates(shell_mesh))
        profile = 2.0 + 0.1 * logical_coords[0]
        component_names = gh_sh_tensor_component_names(
            "SpacetimeMetric", "Pi", "Phi"
        )
        self.gh_h5_filename = os.path.join(self.test_dir, "gh_voldata.h5")
        with spectre_h5.H5File(self.gh_h5_filename, "w") as open_h5file:
            volfile = open_h5file.insert_vol("/GhVolumeData", version=0)
            for observation_id, observation_value in enumerate([0.0, 1.0]):
                tensor_components = [
                    TensorComponent(
                        component_name,
                        (component_index + 1)
                        * (profile + 0.01 * observation_value),
                    )
                    for component_index, component_name in enumerate(
                        component_names
                    )
                ]
                volfile.write_volume_data(
                    observation_id=observation_id,
                    observation_value=observation_value,
                    elements=[
                        ElementVolumeData(
                            ElementId[3](0), tensor_components, shell_mesh
                        )
                    ],
                    serialized_domain=serialize_domain(shell_domain),
                )

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
                "--figsize",
                "12",
                "4",
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
                "--over-time",
                "-o",
                self.plot_filename,
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        # Can't easily test the plot itself, so just check that it was created
        self.assertTrue(os.path.exists(self.plot_filename))
        os.remove(self.plot_filename)

    def test_gh_sh_cli(self):
        runner = CliRunner()
        result = runner.invoke(
            plot_power_monitors_command,
            [
                self.gh_h5_filename,
                "-d",
                "GhVolumeData",
                "--step",
                "-1",
                "-b",
                self.shell_block_name,
                "--gh-sh",
                "--gh-sh-variable",
                "Pi",
                "-o",
                self.plot_filename,
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(os.path.exists(self.plot_filename))
        os.remove(self.plot_filename)

        result = runner.invoke(
            plot_power_monitors_command,
            [
                self.gh_h5_filename,
                "-d",
                "GhVolumeData",
                "-b",
                self.shell_block_name,
                "--gh-sh",
                "--over-time",
            ],
        )
        self.assertEqual(result.exit_code, 2, result.output)
        self.assertIn("--gh-sh-frame-prefix", result.output)

        frame_prefix = os.path.join(self.test_dir, "gh_sh")
        result = runner.invoke(
            plot_power_monitors_command,
            [
                self.gh_h5_filename,
                "-d",
                "GhVolumeData",
                "-b",
                self.shell_block_name,
                "--gh-sh",
                "--over-time",
                "--gh-sh-frame-prefix",
                frame_prefix,
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        frame_files = [
            filename
            for filename in os.listdir(self.test_dir)
            if filename.startswith("gh_sh_") and filename.endswith(".png")
        ]
        self.assertEqual(len(frame_files), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
