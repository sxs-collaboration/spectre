# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import shutil
import unittest

import numpy as np
from click.testing import CliRunner

import spectre.IO.H5 as spectre_h5
from spectre.Informer import unit_test_build_path
from spectre.Visualization.PlotControlSystem import plot_control_system_command


class TestPlotControlSystem(unittest.TestCase):
    def setUp(self):
        self.work_dir = os.path.join(
            unit_test_build_path(), "Visualization/PlotControlSystem"
        )
        self.reductions_file_names = [
            f"TestControlSystemReductions{i}.h5" for i in range(2)
        ]
        system_and_components = {
            "Expansion": ["Expansion"],
            "Translation": ["x", "y", "z"],
            "Rotation": ["x", "y", "z"],
            # These should be ignored
            "SizeA": ["Size"],
            "SizeB": ["Size"],
        }
        for obj in ["A", "B"]:
            system_and_components[f"Shape{obj}"] = []
            for l in range(4):
                for m in range(-l, l + 1):
                    system_and_components[f"Shape{obj}"].append(f"l{l}m{m}")

        shutil.rmtree(self.work_dir, ignore_errors=True)
        os.makedirs(self.work_dir, exist_ok=True)

        # Taken from src/ControlSystem/WriteData.hpp
        legend = [
            "Time",
            "FunctionOfTime",
            "dtFunctionOfTime",
            "d2tFunctionOfTime",
            "ControlError",
            "dtControlError",
            "DampingTimescale",
        ]
        for reduction_file_name in self.reductions_file_names:
            with spectre_h5.H5File(
                os.path.join(self.work_dir, reduction_file_name), "w"
            ) as open_h5_file:
                for system in system_and_components:
                    for component in system_and_components[system]:
                        subfile = open_h5_file.insert_dat(
                            f"ControlSystems/{system}/{component}",
                            legend=legend,
                            version=0,
                        )
                        # The numbers here don't have to be meaningful. We're
                        # just testing the plotting
                        for i in range(4):
                            data = np.random.rand(len(legend))
                            data[0] = float(i)
                            subfile.append(data)
                        open_h5_file.close_current_object()

        self.runner = CliRunner()

    def tearDown(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_plot_size(self):
        # Just one h5 file
        output_file_name = os.path.join(
            self.work_dir, "SingleFileWithShape.pdf"
        )
        result = self.runner.invoke(
            plot_control_system_command,
            [
                os.path.join(self.work_dir, self.reductions_file_names[0]),
                "-o",
                output_file_name,
                "--with-shape",
                "--shape-l_max",
                "3",
                "--show-all-m",
                "--x-bounds",
                "0.0",
                "3.0",
                "--x-label",
                "Time (M)",
                "--title",
                "Control Systems First File",
            ],
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(os.path.exists(output_file_name))

        # Just one h5 file again, but l2 norm of shape coefs
        output_file_name = os.path.join(
            self.work_dir, "SingleFileWithShapeL2Norm.pdf"
        )
        result = self.runner.invoke(
            plot_control_system_command,
            [
                os.path.join(self.work_dir, self.reductions_file_names[0]),
                "-o",
                output_file_name,
                "--with-shape",
                "--shape-l_max",
                "3",
                "--x-bounds",
                "0.0",
                "3.0",
                "--x-label",
                "Time (M)",
                "--title",
                "Control Systems First File",
            ],
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(os.path.exists(output_file_name))

        # Multiple h5 files
        output_file_name = os.path.join(
            self.work_dir, "MultiFileWithoutShape.pdf"
        )
        result = self.runner.invoke(
            plot_control_system_command,
            [
                os.path.join(self.work_dir, self.reductions_file_names[0]),
                os.path.join(self.work_dir, self.reductions_file_names[1]),
                "-o",
                output_file_name,
                "--without-shape",
                "--x-bounds",
                "0.0",
                "3.0",
                "--x-label",
                "Time (M)",
                "--title",
                "Control Systems Both Files",
            ],
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(os.path.exists(output_file_name))


if __name__ == "__main__":
    unittest.main(verbosity=2)
