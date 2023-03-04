#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import shutil
import unittest

import numpy as np
import spectre.IO.H5 as spectre_h5
from click.testing import CliRunner
from spectre.Domain import ElementId, serialize_domain
from spectre.Domain.Creators import Interval
from spectre.Informer import unit_test_build_path
from spectre.IO.H5 import ElementVolumeData, TensorComponent
from spectre.Spectral import Basis, Mesh, Quadrature
from spectre.Visualization.Render1D import render_1d_command


class TestRender1D(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(unit_test_build_path(),
                                     'Visualization/Render1D')
        shutil.rmtree(self.test_dir, ignore_errors=True)
        os.makedirs(self.test_dir, exist_ok=True)

        # Generate 1D volume data
        domain = Interval(lower_x=[0.],
                          upper_x=[1.],
                          initial_refinement_level_x=[1],
                          initial_number_of_grid_points_in_x=[4],
                          is_periodic_in_x=[False]).create_domain()
        serialized_domain = serialize_domain(domain)
        mesh = Mesh[1](4, Basis.Legendre, Quadrature.GaussLobatto)
        self.h5file = os.path.join(self.test_dir, "voldata.h5")
        with spectre_h5.H5File(self.h5file, "w") as open_h5file:
            volfile = open_h5file.insert_vol("/VolumeData", version=0)
            volfile.write_volume_data(
                observation_id=0,
                observation_value=1.,
                elements=[
                    ElementVolumeData(
                        ElementId[1]("[B0,(L1I0)]"),
                        [TensorComponent("U", np.random.rand(4))], mesh),
                    ElementVolumeData(
                        ElementId[1]("[B0,(L1I1)]"),
                        [TensorComponent("U", np.random.rand(4))], mesh),
                ],
                serialized_domain=serialized_domain)
            volfile.write_volume_data(
                observation_id=1,
                observation_value=2.,
                elements=[
                    ElementVolumeData(
                        ElementId[1]("[B0,(L1I0)]"),
                        [TensorComponent("U", np.random.rand(4))], mesh),
                    ElementVolumeData(
                        ElementId[1]("[B0,(L1I1)]"),
                        [TensorComponent("U", np.random.rand(4))], mesh),
                ],
                serialized_domain=serialized_domain)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_render_1d(self):
        # Can't easily test the layout of the plot, so we just test that the
        # script runs without error and produces output.
        # We also don't have ffmpeg installed in the CI container, so we can't
        # test an animation.
        runner = CliRunner()
        plot_file = os.path.join(self.test_dir, "plot")
        result = runner.invoke(
            render_1d_command,
            [self.h5file, "-d", "VolumeData", "--step", "0", "-o", plot_file],
            catch_exceptions=False)
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertTrue(os.path.exists(plot_file + ".pdf"), msg=result.output)


if __name__ == '__main__':
    unittest.main(verbosity=2)
