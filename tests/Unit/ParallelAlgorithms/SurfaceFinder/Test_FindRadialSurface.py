# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import os
import shutil
import unittest

import numpy as np
import numpy.testing as npt
from click.testing import CliRunner

import spectre.IO.H5 as spectre_h5
from spectre.Domain import ElementId, ElementMap, serialize_domain
from spectre.Domain.Creators import Sphere
from spectre.Informer import unit_test_build_path
from spectre.IO.H5.IterElements import Element
from spectre.NumericalAlgorithms.LinearOperators import partial_derivative
from spectre.PointwiseFunctions.AnalyticSolutions.GeneralRelativity import (
    KerrSchild,
)
from spectre.Spectral import Basis, Mesh, Quadrature, logical_coordinates
from spectre.SphericalHarmonics import Strahlkorper, cartesian_coords
from spectre.support.Logging import configure_logging
from spectre.SurfaceFinder.FindRadialSurface import (
    find_radial_surface,
    find_radial_surface_command,
)


def _to_tensor_components(tensors):
    """Convert a dictionary of tensors to a list of tensor components"""
    result = []
    for name, tensor in tensors.items():
        result.extend(
            spectre_h5.TensorComponent(
                name + tensor.component_suffix(i), tensor[i]
            )
            for i in range(len(tensor))
        )
    return result


class TestFindRadialSurface(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(unit_test_build_path(), "SurfaceFinder")
        self.h5_filename = os.path.join(self.test_dir, "TestData.h5")
        self.output_surfaces_filename = os.path.join(
            self.test_dir, "Surfaces.h5"
        )
        shutil.rmtree(self.test_dir, ignore_errors=True)
        os.makedirs(self.test_dir, exist_ok=True)

        # Create a spherical domain
        domain = Sphere(
            inner_radius=1.0,
            outer_radius=3.0,
            excise=True,
            initial_refinement=0,
            initial_number_of_grid_points=8,
            use_equiangular_map=True,
        ).create_domain()
        element_ids = [ElementId[3](block_id) for block_id in range(6)]
        elements = [
            Element(
                id=element_id,
                mesh=Mesh[3](10, Basis.Legendre, Quadrature.GaussLobatto),
                map=ElementMap(element_id, domain),
            )
            for element_id in element_ids
        ]

        # Evaluate a Kerr solution on the domain and write to file
        solution = KerrSchild(mass=1.0, dimensionless_spin=[0.0, 0.0, 0.0])
        tensor_names = ["Lapse"]
        with spectre_h5.H5File(self.h5_filename, "w") as h5file:
            volfile = h5file.insert_vol("element_data", version=0)
            volfile.write_volume_data(
                observation_id=0,
                observation_value=0.0,
                elements=[
                    spectre_h5.ElementVolumeData(
                        element.id,
                        _to_tensor_components(
                            solution.variables(
                                element.inertial_coordinates,
                                tensor_names,
                            )
                        ),
                        element.mesh,
                    )
                    for element in elements
                ],
                serialized_domain=serialize_domain(domain),
            )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_find_radial_surface(self):
        surface = find_radial_surface(
            [self.h5_filename],
            subfile_name="element_data",
            obs_id=0,
            obs_time=0.0,
            var_name="Lapse",
            # Value of the lapse at the horizon
            target=1 / np.sqrt(2),
            initial_guess=Strahlkorper(
                l_max=12, radius=1.0, center=[0.0, 0.0, 0.0]
            ),
            output_surfaces_file=self.output_surfaces_filename,
            output_coeffs_subfile="SurfaceCoeffs",
            output_coords_subfile="SurfaceCoords",
        )
        # Horizon should be a sphere of coordinate radius 2.0
        npt.assert_allclose(
            np.linalg.norm(cartesian_coords(surface), axis=0), 2.0, atol=1e-3
        )

    def test_cli(self):
        runner = CliRunner()
        result = runner.invoke(
            find_radial_surface_command,
            [
                self.h5_filename,
                "-d",
                "element_data",
                "--step",
                "0",
                "-t",
                str(1 / np.sqrt(2)),
                "-y",
                "Lapse",
                "--l-max",
                "12",
                "--initial-radius",
                "1.0",
                "--center",
                "0.0",
                "0.0",
                "0.0",
                "--output-surfaces-file",
                self.output_surfaces_filename,
                "--output-coeffs-subfile",
                "SurfaceCoeffs",
                "--output-coords-subfile",
                "SurfaceCoords",
            ],
            catch_exceptions=True,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(os.path.exists(self.output_surfaces_filename))


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
