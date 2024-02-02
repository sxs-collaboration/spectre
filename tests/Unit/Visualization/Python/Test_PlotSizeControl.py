# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import shutil
import unittest

import numpy as np
from click.testing import CliRunner

import spectre.IO.H5 as spectre_h5
from spectre.Informer import unit_test_build_path
from spectre.Visualization.PlotSizeControl import plot_size_control_command


class TestPlotSizeControl(unittest.TestCase):
    def setUp(self):
        self.work_dir = os.path.join(
            unit_test_build_path(), "Visualization/PlotSize"
        )
        self.reductions_file_names = [
            f"TestSizeReductions{i}.h5" for i in range(2)
        ]
        self.diagnostic_subfile_name = "/ControlSystems/SizeB/Diagnostics.dat"

        shutil.rmtree(self.work_dir, ignore_errors=True)
        os.makedirs(self.work_dir, exist_ok=True)

        # Taken from src/ControlSystem/ControlErrors/Size.cpp
        legend = [
            "Time",
            "ControlError",
            "StateNumber",
            "DiscontinuousChangeHasOccurred",
            "FunctionOfTime",
            "DtFunctionOfTime",
            "HorizonCoef00",
            "AveragedDtHorizonCoef00",
            "RawDtHorizonCoef00",
            "SmootherTimescale",
            "MinDeltaR",
            "MinRelativeDeltaR",
            "AvgDeltaR",
            "AvgRelativeDeltaR",
            "ControlErrorDeltaR",
            "TargetCharSpeed",
            "MinCharSpeed",
            "MinComovingCharSpeed",
            "CharSpeedCrossingTime",
            "ComovingCharSpeedCrossingTime",
            "DeltaRCrossingTime",
            "SuggestedTimescale",
            "DampingTime",
        ]
        for reduction_file_name in self.reductions_file_names:
            with spectre_h5.H5File(
                os.path.join(self.work_dir, reduction_file_name), "w"
            ) as open_h5_file:
                diagnostic_subfile = open_h5_file.insert_dat(
                    self.diagnostic_subfile_name,
                    legend=legend,
                    version=0,
                )
                # The numbers here don't have to be meaningful. We're just
                # testing the plotting
                for i in range(4):
                    data = np.random.rand(len(legend))
                    data[0] = float(i)
                    diagnostic_subfile.append(data)

        self.runner = CliRunner()

    def tearDown(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_plot_size(self):
        # Just one h5 file
        result = self.runner.invoke(
            plot_size_control_command,
            [
                os.path.join(self.work_dir, self.reductions_file_names[0]),
                "-d",
                "B",
                "-o",
                os.path.join(self.work_dir, "SingleFile"),
                "--x-bounds",
                "0.0",
                "3.0",
                "--x-label",
                "Time (M)",
                "--title",
                "Size Control First File",
            ],
            catch_exceptions=False,
        )

        output_file_name = os.path.join(self.work_dir, "SingleFileB.pdf")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(os.path.exists(output_file_name))

        # Multiple h5 files
        result = self.runner.invoke(
            plot_size_control_command,
            [
                os.path.join(self.work_dir, self.reductions_file_names[0]),
                os.path.join(self.work_dir, self.reductions_file_names[1]),
                "-d",
                "B",
                "-o",
                os.path.join(self.work_dir, "MultiFileB"),
                "--x-bounds",
                "0.0",
                "3.0",
                "--x-label",
                "Time (M)",
                "--title",
                "Size Control Both Files",
            ],
            catch_exceptions=False,
        )

        output_file_name = os.path.join(self.work_dir, "MultiFileB.pdf")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(os.path.exists(output_file_name))


if __name__ == "__main__":
    unittest.main(verbosity=2)
