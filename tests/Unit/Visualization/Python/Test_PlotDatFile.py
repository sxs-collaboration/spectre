# Distributed under the MIT License.
# See LICENSE.txt for details.

import spectre.Visualization.PlotDatFile as plot_dat

import spectre.Informer as spectre_informer
import numpy as np
import os
import unittest

# For Py2 compatibility
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


class TestPlotDatFile(unittest.TestCase):
    def test_available_subfiles(self):
        filename = os.path.join(spectre_informer.unit_test_src_path(),
                                'Visualization/Python', 'DatTestData.h5')
        available_subfiles = plot_dat.open_and_process_file(filename, None)
        self.assertEqual(available_subfiles,
                         ['/Group0/MemoryData.dat', '/TimeSteps2.dat'])

    def test_nonexistent_subfile(self):
        filename = os.path.join(spectre_informer.unit_test_src_path(),
                                'Visualization/Python', 'DatTestData.h5')
        with self.assertRaisesRegex(
                Exception, "Unable to open dat file 'TimeSteps.dat'."):
            plot_dat.open_and_process_file(filename, 'TimeSteps')

    def test_legend_only(self):
        filename = os.path.join(spectre_informer.unit_test_src_path(),
                                'Visualization/Python', 'DatTestData.h5')
        legend = plot_dat.open_and_process_file(filename,
                                                'TimeSteps2',
                                                legend_only=True,
                                                write_dat=None)
        self.assertEqual(legend, [
            'Time', 'NumberOfPoints', 'Slab size', 'Minimum time step',
            'Maximum time step', 'Effective time step', 'Minimum Walltime',
            'Maximum Walltime'
        ])

    def test_write_dat(self):
        filename = os.path.join(spectre_informer.unit_test_src_path(),
                                'Visualization/Python', 'DatTestData.h5')
        dat_filename = os.path.join(spectre_informer.unit_test_build_path(),
                                    'Visualization/Python', 'DatTestData.dat')
        plot_dat.open_and_process_file(filename,
                                       'TimeSteps2',
                                       legend_only=False,
                                       write_dat=dat_filename)
        # Load written out dat file and compare to expected values
        self.assertTrue(os.path.exists(dat_filename))
        dat_data = np.loadtxt(dat_filename)
        self.assertTrue(
            np.allclose(dat_data,
                        np.asarray([[
                            0.0, 20.0, 1.0e-2, 6.25e-4, 6.25e-4, 6.25e-4,
                            3.3872221e-2, 3.3873593e-2
                        ],
                                    [
                                        1.0e-2, 20.0, 1.0e-2, 5.0e-3, 5.0e-3,
                                        5.0e-3, 2.75874901e-01,
                                        2.7600519899999998e-01
                                    ]]),
                        rtol=1.0e-12,
                        atol=1.0e-20))
        if os.path.exists(dat_filename):
            os.remove(dat_filename)

    def test_write_plot(self):
        filename = os.path.join(spectre_informer.unit_test_src_path(),
                                'Visualization/Python', 'DatTestData.h5')
        plot_filename = os.path.join(spectre_informer.unit_test_build_path(),
                                     'Visualization/Python', 'DatTestData')
        plot_dat.open_and_process_file(filename,
                                       'TimeSteps2',
                                       legend_only=False,
                                       write_dat=None,
                                       x_axis=None,
                                       functions=['Slab size'],
                                       linestyles=['-'],
                                       labels=None,
                                       linewidth=1.0,
                                       y_logscale=False,
                                       x_label=None,
                                       fontsize=16,
                                       legend_ncols=1,
                                       x_bounds=None,
                                       y_bounds=None,
                                       y_label="Blah",
                                       title="DatTestdata",
                                       output=plot_filename)
        # We can't test that the plot is correct, so just make sure it at least
        # gets written out.
        self.assertTrue(os.path.exists(plot_filename + '.pdf'))
        if os.path.exists(plot_filename + '.pdf'):
            os.remove(plot_filename + '.pdf')

        # Test that no y-label doesn't error
        plot_dat.open_and_process_file(filename,
                                       'TimeSteps2',
                                       legend_only=False,
                                       write_dat=None,
                                       x_axis=None,
                                       functions=['Slab size'],
                                       linestyles=['-'],
                                       labels=None,
                                       linewidth=1.0,
                                       y_logscale=False,
                                       x_label=None,
                                       fontsize=16,
                                       legend_ncols=1,
                                       x_bounds=None,
                                       y_bounds=None,
                                       y_label=None,
                                       title="DatTestdata",
                                       output=plot_filename)
        # We can't test that the plot is correct, so just make sure it at least
        # gets written out.
        self.assertTrue(os.path.exists(plot_filename + '.pdf'))
        if os.path.exists(plot_filename + '.pdf'):
            os.remove(plot_filename + '.pdf')

        # Test that x and y bounds don't error
        plot_dat.open_and_process_file(filename,
                                       'TimeSteps2',
                                       legend_only=False,
                                       write_dat=None,
                                       x_axis=None,
                                       functions=['Slab size'],
                                       linestyles=['-'],
                                       labels=None,
                                       linewidth=1.0,
                                       y_logscale=False,
                                       x_label=None,
                                       fontsize=16,
                                       legend_ncols=1,
                                       x_bounds=[0.0, 1.0],
                                       y_bounds=[0.0, 1.0],
                                       y_label=None,
                                       title="DatTestdata",
                                       output=plot_filename)
        # We can't test that the plot is correct, so just make sure it at least
        # gets written out.
        self.assertTrue(os.path.exists(plot_filename + '.pdf'))
        if os.path.exists(plot_filename + '.pdf'):
            os.remove(plot_filename + '.pdf')

        # Test that x-label doesn't error
        plot_dat.open_and_process_file(filename,
                                       'TimeSteps2',
                                       legend_only=False,
                                       write_dat=None,
                                       x_axis=None,
                                       functions=['Slab size'],
                                       linestyles=['-'],
                                       labels=None,
                                       linewidth=1.0,
                                       y_logscale=False,
                                       x_label="X label",
                                       fontsize=16,
                                       legend_ncols=1,
                                       x_bounds=None,
                                       y_bounds=None,
                                       y_label=None,
                                       title="DatTestdata",
                                       output=plot_filename)
        # We can't test that the plot is correct, so just make sure it at least
        # gets written out.
        self.assertTrue(os.path.exists(plot_filename + '.pdf'))
        if os.path.exists(plot_filename + '.pdf'):
            os.remove(plot_filename + '.pdf')

        # Test that y logscale doesn't error
        plot_dat.open_and_process_file(filename,
                                       'TimeSteps2',
                                       legend_only=False,
                                       write_dat=None,
                                       x_axis=None,
                                       functions=['Slab size'],
                                       linestyles=['-'],
                                       labels=None,
                                       linewidth=1.0,
                                       y_logscale=True,
                                       x_label=None,
                                       fontsize=16,
                                       legend_ncols=1,
                                       x_bounds=None,
                                       y_bounds=None,
                                       y_label=None,
                                       title="DatTestdata",
                                       output=plot_filename)
        # We can't test that the plot is correct, so just make sure it at least
        # gets written out.
        self.assertTrue(os.path.exists(plot_filename + '.pdf'))
        if os.path.exists(plot_filename + '.pdf'):
            os.remove(plot_filename + '.pdf')

        # Test that the x-axis can be set to something different without error
        plot_dat.open_and_process_file(filename,
                                       'TimeSteps2',
                                       legend_only=False,
                                       write_dat=None,
                                       x_axis="NumberOfPoints",
                                       functions=['Slab size'],
                                       linestyles=['-'],
                                       labels=None,
                                       linewidth=1.0,
                                       y_logscale=False,
                                       x_label=None,
                                       fontsize=16,
                                       legend_ncols=1,
                                       x_bounds=None,
                                       y_bounds=None,
                                       y_label=None,
                                       title="DatTestdata",
                                       output=plot_filename)
        # We can't test that the plot is correct, so just make sure it at least
        # gets written out.
        self.assertTrue(os.path.exists(plot_filename + '.pdf'))
        if os.path.exists(plot_filename + '.pdf'):
            os.remove(plot_filename + '.pdf')

        # Test that groups can be accessed without error
        plot_dat.open_and_process_file(filename,
                                       'Group0/MemoryData',
                                       legend_only=False,
                                       write_dat=None,
                                       x_axis=None,
                                       functions=['Usage in MB'],
                                       linestyles=['-'],
                                       labels=None,
                                       linewidth=1.0,
                                       y_logscale=False,
                                       x_label=None,
                                       fontsize=16,
                                       legend_ncols=1,
                                       x_bounds=None,
                                       y_bounds=None,
                                       y_label=None,
                                       title="DatTestdata",
                                       output=plot_filename)
        # We can't test that the plot is correct, so just make sure it at least
        # gets written out.
        self.assertTrue(os.path.exists(plot_filename + '.pdf'))
        if os.path.exists(plot_filename + '.pdf'):
            os.remove(plot_filename + '.pdf')


if __name__ == '__main__':
    unittest.main(verbosity=2)
