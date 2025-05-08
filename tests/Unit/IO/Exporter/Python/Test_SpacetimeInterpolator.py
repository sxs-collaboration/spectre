# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import shutil
import unittest

import numpy as np
import numpy.testing as npt
from click.testing import CliRunner

import spectre.IO.H5 as spectre_h5
from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import Frame, Scalar, tnsr
from spectre.Domain import ElementId, serialize_domain
from spectre.Domain.Creators import Brick
from spectre.Informer import unit_test_build_path, unit_test_src_path
from spectre.IO.Exporter import ObservationStep, SpacetimeInterpolator
from spectre.IO.H5 import ElementVolumeData, TensorComponent
from spectre.Spectral import Basis, Mesh, Quadrature


class TestSpacetimeInterpolator(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(
            unit_test_build_path(), "IO/Exporter/SpacetimeInterpolator"
        )
        self.h5_filename = os.path.join(self.test_dir, "VolumeData.h5")
        os.makedirs(self.test_dir, exist_ok=True)

        # Write some volume data
        domain = Brick(
            lower_bounds=[0.0, 0.0, 0.0],
            upper_bounds=[1.0, 1.0, 1.0],
            initial_refinement_levels=[0, 0, 0],
            initial_num_points=[4, 4, 4],
            is_periodic=[False, False, False],
        ).create_domain()
        serialized_domain = serialize_domain(domain)
        mesh = Mesh[3](4, Basis.Legendre, Quadrature.GaussLobatto)
        with spectre_h5.H5File(self.h5_filename, "w") as open_h5file:
            volfile = open_h5file.insert_vol("/VolumeData", version=0)
            for i in range(5):
                volfile.write_volume_data(
                    observation_id=i,
                    observation_value=i,
                    elements=[
                        ElementVolumeData(
                            ElementId[3]("[B0,(L0I0,L0I0,L0I0)]"),
                            [TensorComponent("Psi", np.ones(4**3) * i)],
                            mesh,
                        ),
                    ],
                    serialized_domain=serialized_domain,
                )

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_spacetime_interpolator(self):
        interpolator = SpacetimeInterpolator[3](
            self.h5_filename,
            subfile_name="VolumeData",
            tensor_components=["Psi"],
        )
        self.assertEqual(interpolator.max_time_bounds(), [1, 3])
        interpolator.load_time_bounds([1.5, 3])
        self.assertEqual(interpolator.time_bounds(), [1, 3])
        (psi,) = interpolator.interpolate_to_point(np.zeros(3), time=2.5)
        self.assertAlmostEqual(psi, 2.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
