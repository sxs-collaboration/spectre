# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import shutil
import unittest

import h5py
import numpy as np
from click.testing import CliRunner

import spectre.IO.H5 as spectre_h5
from spectre.Informer import unit_test_build_path
from spectre.Visualization.PlotCce import plot_cce_command


class TestPlotCce(unittest.TestCase):
    def setUp(self):
        self.work_dir = os.path.join(
            unit_test_build_path(), "Visualization/PlotCce"
        )
        self.reduction_file_name_one_subfile = "TestCceReductions1.h5"
        self.reduction_file_name_two_subfiles = "TestCceReductions2.h5"
        plot_quantities = [
            "EthInertialRetardedTime",
            "Strain",
            "News",
            "Psi0",
            "Psi1",
            "Psi2",
            "Psi3",
            "Psi4",
        ]
        self.l_max = 2
        self.extraction_radius_1 = 150
        self.extraction_radius_2 = 250

        shutil.rmtree(self.work_dir, ignore_errors=True)
        os.makedirs(self.work_dir, exist_ok=True)

        legend = [
            "time",
            "Real Y_0,0",
            "Imag Y_0,0",
            "Real Y_1,-1",
            "Imag Y_1,-1",
            "Real Y_1,0",
            "Imag Y_1,0",
            "Real Y_1,1",
            "Imag Y_1,1",
            "Real Y_2,-2",
            "Imag Y_2,-2",
            "Real Y_2,-1",
            "Imag Y_2,-1",
            "Real Y_2,0",
            "Imag Y_2,0",
            "Real Y_2,1",
            "Imag Y_2,1",
            "Real Y_2,2",
            "Imag Y_2,2",
        ]

        self.cce_data = {}
        for quantity in plot_quantities:
            self.cce_data[quantity] = [np.random.rand(len(legend))]
            self.cce_data[quantity].append(np.random.rand(len(legend)))
            for i in range(2):
                self.cce_data[quantity][i][0] = float(i)

        with spectre_h5.H5File(
            os.path.join(self.work_dir, self.reduction_file_name_one_subfile),
            "w",
        ) as open_h5_file:
            cce_subfile = open_h5_file.insert_cce(
                path=f"SpectreR{self.extraction_radius_1:04}",
                l_max=self.l_max,
                version=0,
            )
            # Doesn't matter that we are repeating data. Just testing that
            # things get plotted
            cce_subfile.append(
                {
                    quantity: self.cce_data[quantity][0]
                    for quantity in plot_quantities
                }
            )
            cce_subfile.append(
                {
                    quantity: self.cce_data[quantity][1]
                    for quantity in plot_quantities
                }
            )
            open_h5_file.close_current_object()
            for quantity in plot_quantities:
                dat_subfile = open_h5_file.insert_dat(
                    path=f"Cce/{quantity}", legend=legend, version=0
                )
                dat_subfile.append(self.cce_data[quantity][0])
                dat_subfile.append(self.cce_data[quantity][1])
                open_h5_file.close_current_object()
        with spectre_h5.H5File(
            os.path.join(self.work_dir, self.reduction_file_name_two_subfiles),
            "w",
        ) as open_h5_file:
            cce_subfile = open_h5_file.insert_cce(
                path=f"SpectreR{self.extraction_radius_1:04}",
                l_max=self.l_max,
                version=0,
            )
            # Doesn't matter that we are repeating data. Just testing that
            # things get plotted
            cce_subfile.append(
                {
                    quantity: self.cce_data[quantity][0]
                    for quantity in plot_quantities
                }
            )
            cce_subfile.append(
                {
                    quantity: self.cce_data[quantity][1]
                    for quantity in plot_quantities
                }
            )
            open_h5_file.close_current_object()
            cce_subfile = open_h5_file.insert_cce(
                path=f"SpectreR{self.extraction_radius_2:04}",
                l_max=self.l_max,
                version=0,
            )
            # Doesn't matter that we are repeating data. Just testing that
            # things get plotted
            cce_subfile.append(
                {
                    quantity: self.cce_data[quantity][0]
                    for quantity in plot_quantities
                }
            )
            cce_subfile.append(
                {
                    quantity: self.cce_data[quantity][1]
                    for quantity in plot_quantities
                }
            )

        self.runner = CliRunner()

    def tearDown(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_plot_cce(self):
        output_file_name = os.path.join(self.work_dir, "PlotCceTest1.pdf")
        result = self.runner.invoke(
            plot_cce_command,
            [
                os.path.join(
                    self.work_dir, self.reduction_file_name_one_subfile
                ),
                "-o",
                output_file_name,
                "-m",
                "2,2",
                "-m",
                "2,-2",
                "-m",
                "2,0",
                "-m",
                "1,0",
                "--imag",
                "--x-bounds",
                "0.0",
                "3.0",
                "--x-label",
                "Retarded Time (M)",
                "--title",
                "Cce at Scri Test1",
            ],
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0, result.output)
        print(result.output)
        self.assertTrue(os.path.exists(output_file_name))

        output_file_name = os.path.join(self.work_dir, "PlotCceTest2.pdf")
        result = self.runner.invoke(
            plot_cce_command,
            [
                os.path.join(
                    self.work_dir, self.reduction_file_name_two_subfiles
                ),
                "-o",
                output_file_name,
                "-m",
                "2,2",
                "--real",
                "--extraction-radius",
                "250",
                "--x-bounds",
                "0.0",
                "3.0",
                "--x-label",
                "Retarded Time (M)",
                "--title",
                "Cce at Scri Test2",
            ],
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(os.path.exists(output_file_name))

        output_file_name = os.path.join(self.work_dir, "PlotCceTest3.pdf")
        result = self.runner.invoke(
            plot_cce_command,
            [
                os.path.join(
                    self.work_dir, self.reduction_file_name_one_subfile
                ),
                "-o",
                output_file_name,
                "-m",
                "2,2",
                "--cce-group",
                "Cce",
                "--x-bounds",
                "0.0",
                "3.0",
                "--x-label",
                "Retarded Time (M)",
                "--title",
                "Cce at Scri Test3",
            ],
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(os.path.exists(output_file_name))

        result = self.runner.invoke(
            plot_cce_command,
            [
                os.path.join(
                    self.work_dir, self.reduction_file_name_two_subfiles
                ),
                "-l",
                "-m",
                "2,2",
                "--x-bounds",
                "0.0",
                "3.0",
                "--x-label",
                "Retarded Time (M)",
                "--title",
                "Cce at Scri List",
            ],
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn(
            f"SpectreR{self.extraction_radius_1:04}.cce", result.output
        )
        self.assertIn(
            f"SpectreR{self.extraction_radius_2:04}.cce", result.output
        )

    def test_plot_cce_errors(self):
        error_filename = os.path.join(self.work_dir, "PlotCceTestError.pdf")
        result = self.runner.invoke(
            plot_cce_command,
            [
                os.path.join(
                    self.work_dir, self.reduction_file_name_one_subfile
                ),
                "-o",
                error_filename,
                "-m",
                "2,2",
                "--imag",
                "--real",
                "--x-bounds",
                "0.0",
                "3.0",
                "--x-label",
                "Retarded Time (M)",
                "--title",
                "Cce at Scri Error1",
            ],
            catch_exceptions=False,
        )

        self.assertNotEqual(result.exit_code, 0, result.output)
        self.assertIn("Only specify one of '--real'/'--imag'.", result.output)

        result = self.runner.invoke(
            plot_cce_command,
            [
                os.path.join(
                    self.work_dir, self.reduction_file_name_one_subfile
                ),
                "-o",
                error_filename,
                "-m",
                "2,2",
                "-r",
                "1000",
                "--x-bounds",
                "0.0",
                "3.0",
                "--x-label",
                "Retarded Time (M)",
                "--title",
                "Cce at Scri Error2",
            ],
            catch_exceptions=False,
        )

        self.assertNotEqual(result.exit_code, 0, result.output)
        self.assertIn(
            (
                "Either specify the correct extraction radius, or remove the"
                " option altogether."
            ),
            result.output,
        )

        result = self.runner.invoke(
            plot_cce_command,
            [
                os.path.join(
                    self.work_dir, self.reduction_file_name_two_subfiles
                ),
                "-o",
                error_filename,
                "-m",
                "2,2",
                "--x-bounds",
                "0.0",
                "3.0",
                "--x-label",
                "Retarded Time (M)",
                "--title",
                "Cce at Scri Error2",
            ],
            catch_exceptions=False,
        )

        self.assertNotEqual(result.exit_code, 0, result.output)
        self.assertIn("you did not specify an extraction radius", result.output)

        bad_h5_file_name = "NoCceSubfile.h5"
        shutil.copy(
            os.path.join(self.work_dir, self.reduction_file_name_one_subfile),
            os.path.join(self.work_dir, bad_h5_file_name),
        )
        with h5py.File(
            os.path.join(self.work_dir, bad_h5_file_name), "a"
        ) as h5_file:
            del h5_file[f"SpectreR{self.extraction_radius_1:04}.cce"]

        result = self.runner.invoke(
            plot_cce_command,
            [
                os.path.join(self.work_dir, bad_h5_file_name),
                "-o",
                error_filename,
                "-m",
                "2,2",
                "--x-bounds",
                "0.0",
                "3.0",
                "--x-label",
                "Retarded Time (M)",
                "--title",
                "Cce at Scri Error3",
            ],
            catch_exceptions=False,
        )

        self.assertNotEqual(result.exit_code, 0, result.output)
        self.assertIn(
            "Could not find any Cce subfiles in H5 file", result.output
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
