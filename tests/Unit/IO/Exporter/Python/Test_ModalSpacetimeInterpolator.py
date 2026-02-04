# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import shutil
import unittest

import numpy as np

import spectre.IO.H5 as spectre_h5
from spectre.Domain import ElementId, serialize_domain
from spectre.Domain.Creators import Interval
from spectre.Informer import unit_test_build_path
from spectre.IO.Exporter import ModalSpacetimeInterpolator
from spectre.IO.H5 import ElementVolumeData, TensorComponent
from spectre.Spectral import Basis, Mesh, Quadrature


class TestModalSpacetimeInterpolator(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(
            unit_test_build_path(), "IO/Exporter/ModalSpacetimeInterpolator"
        )
        self.h5_filename = os.path.join(self.test_dir, "VolumeData.h5")
        os.makedirs(self.test_dir, exist_ok=True)

        domain = Interval(
            lower_bounds=[0.0],
            upper_bounds=[1.0],
            initial_refinement_levels=[0],
            initial_num_points=[4],
            is_periodic=[False],
        ).create_domain()
        serialized_domain = serialize_domain(domain)
        mesh = Mesh[1](4, Basis.Legendre, Quadrature.GaussLobatto)
        num_points = mesh.number_of_grid_points()
        with spectre_h5.H5File(self.h5_filename, "w") as open_h5file:
            volfile = open_h5file.insert_vol("/VolumeData", version=0)
            for i in range(10):
                volfile.write_volume_data(
                    observation_id=i,
                    observation_value=float(i),
                    elements=[
                        ElementVolumeData(
                            ElementId[1]("[B0,(L0I0)]"),
                            [TensorComponent("Psi", np.ones(num_points) * i)],
                            mesh,
                        ),
                    ],
                    serialized_domain=serialized_domain,
                )

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_modal_spacetime_interpolator(self):
        interpolator = ModalSpacetimeInterpolator[1](
            self.h5_filename,
            subfiles_coarsest_to_finest=["VolumeData"],
            tensor_components=["Psi"],
        )
        (psi,) = interpolator.interpolate_to_point(np.array([0.2]), time=2.5)
        self.assertAlmostEqual(psi, 2.5)

        serialized_filename = os.path.join(self.test_dir, "Interpolator.h5")
        interpolator.write_to_h5(
            serialized_filename, "/ModalSpacetimeInterpolator"
        )
        reloaded = ModalSpacetimeInterpolator[1](
            serialized_filename, "/ModalSpacetimeInterpolator"
        )
        (psi_reloaded,) = reloaded.interpolate_to_point(
            np.array([0.2]), time=2.5
        )
        self.assertAlmostEqual(psi_reloaded, 2.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
