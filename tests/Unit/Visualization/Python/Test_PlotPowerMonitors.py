# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import shutil
import unittest

import matplotlib.pyplot as plt
import numpy as np
from click.testing import CliRunner

import spectre.IO.H5 as spectre_h5
from spectre.Domain import ElementId, deserialize_domain, serialize_domain
from spectre.Domain.Creators import Cylinder, SphericalShells
from spectre.Informer import unit_test_build_path, unit_test_src_path
from spectre.IO.H5 import ElementVolumeData, TensorComponent, open_volfiles
from spectre.Spectral import Basis, Mesh, Quadrature, logical_coordinates
from spectre.Visualization.PlotPowerMonitors import (
    _SH_SYSTEMS,
    _detect_sh_system,
    find_block_or_group,
    gh_sh_tensor_component_names,
    plot_power_monitors_command,
    plot_sw_power_monitors,
    sw_sh_tensor_component_names,
)


class FakeH5File:
    """Stand-in for `spectre_h5.H5File` that only serves an input source."""

    def __init__(self, input_source: str):
        self._input_source = input_source

    def input_source(self):
        return self._input_source


class FakeVolfile:
    """Stand-in for `spectre_h5.H5Vol` that only serves component names."""

    def __init__(self, component_names):
        self._component_names = component_names

    def list_observation_ids(self):
        return [] if self._component_names is None else [0]

    def list_tensor_components(self, observation_id):
        return self._component_names


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
        self.shell_mesh = shell_mesh
        self.shell_domain = shell_domain

        self.gh_h5_filename = os.path.join(self.test_dir, "gh_voldata.h5")
        self.write_volume_file(
            self.gh_h5_filename,
            "/GhVolumeData",
            gh_sh_tensor_component_names("SpacetimeMetric", "Pi", "Phi"),
        )
        self.sw_h5_filename = os.path.join(self.test_dir, "sw_voldata.h5")
        self.write_volume_file(
            self.sw_h5_filename,
            "/SwVolumeData",
            sw_sh_tensor_component_names("Psi", "Pi", "Phi"),
        )

        # A filled-sphere (ZernikeB3) element next to a Legendre one, to cover
        # the CLI's basis dispatch: the B3 element must reach the filled-sphere
        # monitors and the Legendre element must be skipped
        self.b3_n_r = 4
        self.b3_l_max = 2
        self.b3_mesh = Mesh[3](
            [self.b3_n_r, self.b3_l_max + 1, 2 * self.b3_l_max + 1],
            [Basis.ZernikeB3, Basis.ZernikeB3, Basis.ZernikeB3],
            [
                Quadrature.GaussRadauUpper,
                Quadrature.Gauss,
                Quadrature.Equiangular,
            ],
        )
        legendre_mesh = Mesh[3](
            [3, 3, 3],
            [Basis.Legendre, Basis.Legendre, Basis.Legendre],
            [
                Quadrature.GaussLobatto,
                Quadrature.GaussLobatto,
                Quadrature.GaussLobatto,
            ],
        )
        self.b3_h5_filename = os.path.join(self.test_dir, "b3_voldata.h5")
        self.write_volume_file(
            self.b3_h5_filename,
            "/SwVolumeData",
            sw_sh_tensor_component_names("Psi", "Pi", "Phi"),
            meshes=[self.b3_mesh, legendre_mesh],
        )

    def write_volume_file(
        self, h5_filename, subfile_name, component_names, meshes=None
    ):
        """Write two observations of one element per entry in 'meshes'.

        Defaults to a single element on the spherical-shell mesh. Passing
        several meshes puts elements of different bases in one file, which
        exercises the CLI's per-element basis dispatch.
        """
        if meshes is None:
            meshes = [self.shell_mesh]
        # One element per mesh, all in block 0, refined just enough to give
        # each element a distinct id.
        refinement = max(len(meshes) - 1, 0).bit_length()
        element_ids = [
            ElementId[3](f"[B0,(L{refinement}I{i},L0I0,L0I0)]")
            for i in range(len(meshes))
        ]
        with spectre_h5.H5File(h5_filename, "w") as open_h5file:
            volfile = open_h5file.insert_vol(subfile_name, version=0)
            for observation_id, observation_value in enumerate([0.0, 1.0]):
                elements = []
                for element_id, mesh in zip(element_ids, meshes):
                    profile = (
                        2.0 + 0.1 * np.asarray(logical_coordinates(mesh))[0]
                    )
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
                    elements.append(
                        ElementVolumeData(element_id, tensor_components, mesh)
                    )
                volfile.write_volume_data(
                    observation_id=observation_id,
                    observation_value=observation_value,
                    elements=elements,
                    serialized_domain=serialize_domain(self.shell_domain),
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
                "--sh",
                "--sh-variable",
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
                "--sh",
                "--over-time",
            ],
        )
        self.assertEqual(result.exit_code, 2, result.output)
        self.assertIn("--sh-frame-prefix", result.output)

        frame_prefix = os.path.join(self.test_dir, "gh_sh")
        result = runner.invoke(
            plot_power_monitors_command,
            [
                self.gh_h5_filename,
                "-d",
                "GhVolumeData",
                "-b",
                self.shell_block_name,
                "--sh",
                "--over-time",
                "--sh-frame-prefix",
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

    def test_sw_sh_tensor_component_names(self):
        self.assertEqual(
            sw_sh_tensor_component_names("Psi", "Pi", "Phi"),
            ["Psi", "Pi", "Phi_x", "Phi_y", "Phi_z"],
        )

    def test_detect_sh_system(self):
        # Detection from the 'Executable' field of the input source
        self.assertEqual(
            _detect_sh_system(
                FakeH5File("Executable: EvolveScalarWave3D\n"),
                FakeVolfile(None),
            ),
            "sw",
        )
        self.assertEqual(
            _detect_sh_system(
                FakeH5File("Executable: EvolveGhSingleBlackHole\n"),
                FakeVolfile(None),
            ),
            "gh",
        )
        # CurvedScalarWave executables must not be mistaken for ScalarWave
        for executable in (
            "EvolveCurvedScalarWaveMinkowski3D",
            "EvolveWorldtubeCurvedScalarWaveKerrSchild3D",
        ):
            self.assertEqual(
                _detect_sh_system(
                    FakeH5File(f"Executable: {executable}\n"),
                    FakeVolfile(None),
                ),
                "csw",
            )
        # GhValenciaDivClean is not a plain GH system, so it falls back to the
        # tensor components
        self.assertEqual(
            _detect_sh_system(
                FakeH5File("Executable: EvolveGhValenciaDivCleanBns\n"),
                FakeVolfile(["Psi", "Pi", "Phi_x"]),
            ),
            "sw",
        )
        # Detection from the tensor components when there is no input source.
        # ScalarWave and CurvedScalarWave share these component names, so the
        # fallback reports 'sw' for both; the monitors are identical either
        # way.
        self.assertEqual(
            _detect_sh_system(
                FakeH5File(""),
                FakeVolfile(sw_sh_tensor_component_names("Psi", "Pi", "Phi")),
            ),
            "sw",
        )
        self.assertEqual(
            _detect_sh_system(
                FakeH5File(""),
                FakeVolfile(
                    gh_sh_tensor_component_names("SpacetimeMetric", "Pi", "Phi")
                ),
            ),
            "gh",
        )
        # Neither source of information is conclusive
        with self.assertRaisesRegex(Exception, "--sh-system"):
            _detect_sh_system(FakeH5File(""), FakeVolfile(["Lapse", "Shift_x"]))

    def test_sw_sh_cli(self):
        runner = CliRunner()
        # The system is auto-detected as 'sw' from the tensor components.
        # '--sh-variable' is repeatable, and '--fixed-y-limits' is accepted on
        # the single-observation path as well as for movie frames.
        result = runner.invoke(
            plot_power_monitors_command,
            [
                self.sw_h5_filename,
                "-d",
                "SwVolumeData",
                "--step",
                "-1",
                "-b",
                self.shell_block_name,
                "--sh",
                "--sh-variable",
                "Psi",
                "--sh-variable",
                "Pi",
                "--fixed-y-limits",
                "1e-12",
                "1e2",
                "-o",
                self.plot_filename,
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(os.path.exists(self.plot_filename))
        os.remove(self.plot_filename)

        # Variables of another system are rejected for the detected system
        result = runner.invoke(
            plot_power_monitors_command,
            [
                self.sw_h5_filename,
                "-d",
                "SwVolumeData",
                "--step",
                "-1",
                "-b",
                self.shell_block_name,
                "--sh",
                "--sh-variable",
                "SpacetimeMetric",
                "-o",
                self.plot_filename,
            ],
        )
        self.assertEqual(result.exit_code, 2, result.output)
        self.assertIn("SpacetimeMetric", result.output)

        # All variables over time, one frame per observation
        frame_prefix = os.path.join(self.test_dir, "sw_sh")
        result = runner.invoke(
            plot_power_monitors_command,
            [
                self.sw_h5_filename,
                "-d",
                "SwVolumeData",
                "-b",
                self.shell_block_name,
                "--sh",
                "--sh-system",
                "sw",
                "--over-time",
                "--sh-frame-prefix",
                frame_prefix,
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        frame_files = [
            filename
            for filename in os.listdir(self.test_dir)
            if filename.startswith("sw_sh_") and filename.endswith(".png")
        ]
        self.assertEqual(len(frame_files), 2)

    def test_csw_sh_cli(self):
        """'--sh-system csw' reuses the ScalarWave monitors unchanged."""
        # CurvedScalarWave evolves the same variables as ScalarWave, which is
        # what makes reusing 'plot_sw_power_monitors' for 'csw' valid
        self.assertEqual(
            _SH_SYSTEMS["csw"]["variables"], _SH_SYSTEMS["sw"]["variables"]
        )

        # Restrict to a single variable so only one column is rendered; this
        # test only needs to cover the 'csw' dispatch, and the full 2x3 grid
        # is the expensive part of these CLI tests
        runner = CliRunner()
        result = runner.invoke(
            plot_power_monitors_command,
            [
                self.sw_h5_filename,
                "-d",
                "SwVolumeData",
                "--step",
                "-1",
                "-b",
                self.shell_block_name,
                "--sh",
                "--sh-system",
                "csw",
                "--sh-variable",
                "Psi",
                "-o",
                self.plot_filename,
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(os.path.exists(self.plot_filename))
        os.remove(self.plot_filename)

    def test_b3_and_skipped_element(self):
        """ZernikeB3 elements reach the filled-sphere monitors, and elements
        with no spherical-harmonic basis are skipped instead of erroring."""
        with spectre_h5.H5File(self.b3_h5_filename, "r") as open_h5file:
            volfile = open_h5file.get_vol("/SwVolumeData")
            domain = deserialize_domain[3](volfile.get_domain())
            obs_id = volfile.list_observation_ids()[0]

        figure = plot_sw_power_monitors(
            open_volfiles([self.b3_h5_filename], "SwVolumeData", obs_id),
            obs_id=obs_id,
            block_or_group_names=[self.shell_block_name],
            domain=domain,
            variables_to_plot=("Psi",),
        )
        # Exactly one line per row: the ZernikeB3 element is plotted and the
        # Legendre element is skipped by 'get_monitors' returning None. A
        # broken B3 dispatch would skip both and leave the axes empty.
        radial_axes, angular_axes = figure.get_axes()
        self.assertEqual(len(radial_axes.get_lines()), 1)
        self.assertEqual(len(angular_axes.get_lines()), 1)
        # The filled-sphere radial monitor has n_r entries and the angular
        # monitor has l_max + 1.
        self.assertEqual(
            len(radial_axes.get_lines()[0].get_ydata()), self.b3_n_r
        )
        self.assertEqual(
            len(angular_axes.get_lines()[0].get_ydata()), self.b3_l_max + 1
        )
        plt.close(figure)


if __name__ == "__main__":
    unittest.main(verbosity=2)
