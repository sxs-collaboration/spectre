# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import shutil
import unittest

import numpy as np
from click.testing import CliRunner

import spectre.IO.H5 as spectre_h5
from spectre.Informer import unit_test_build_path
from spectre.Visualization.PlotMemoryMonitors import (
    plot_memory_monitors_command,
)


class TestPlotMemoryMonitors(unittest.TestCase):
    def setUp(self):
        self.work_dir = os.path.join(
            unit_test_build_path(), "Visualization/PlotMemory"
        )
        self.reductions_file_names = [
            f"TestMemoryReductions{i}.h5" for i in range(2)
        ]
        self.memory_monitor_dirs = "/MemoryMonitors"

        shutil.rmtree(self.work_dir, ignore_errors=True)
        os.makedirs(self.work_dir, exist_ok=True)

        singleton_legend = ["Time", "Proc", "Size (MB)"]
        nodegroup_array_legend = [
            "Time",
            "Size on node 0 (MB)",
            "Size on node 1 (MB)",
            "Average size per node (MB)",
        ]
        group_legend = [
            "Time",
            "Size on node 0 (MB)",
            "Size on node 1 (MB)",
            "Proc of max size",
            "Size on proc of max size (MB)",
            "Average size per node (MB)",
        ]
        filenames = ["Singleton.dat", "Nodegroup.dat", "Array.dat", "Group.dat"]
        legends = [
            singleton_legend,
            nodegroup_array_legend,
            nodegroup_array_legend,
            group_legend,
        ]

        for reduction_file_name in self.reductions_file_names:
            with spectre_h5.H5File(
                os.path.join(self.work_dir, reduction_file_name), "w"
            ) as open_h5_file:
                for filename, legend in zip(filenames, legends):
                    subfile = open_h5_file.insert_dat(
                        f"{self.memory_monitor_dirs}/{filename}",
                        legend=legend,
                        version=0,
                    )
                    # The numbers here don't have to be meaningful. We're just
                    # testing the plotting
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
        output_file_name = os.path.join(self.work_dir, "SingleFile.pdf")
        result = self.runner.invoke(
            plot_memory_monitors_command,
            [
                os.path.join(self.work_dir, self.reductions_file_names[0]),
                "-o",
                output_file_name,
                "--x-bounds",
                "0.0",
                "3.0",
                "--use-mb",
                "--x-label",
                "Time (M)",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(os.path.exists(output_file_name))

        # Multiple h5 files
        output_file_name = os.path.join(self.work_dir, "MultiFile.pdf")
        result = self.runner.invoke(
            plot_memory_monitors_command,
            [
                os.path.join(self.work_dir, self.reductions_file_names[0]),
                os.path.join(self.work_dir, self.reductions_file_names[1]),
                "-o",
                output_file_name,
                "--x-label",
                "Time (M)",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(os.path.exists(output_file_name))


if __name__ == "__main__":
    unittest.main(verbosity=2)
