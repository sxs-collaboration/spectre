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
from spectre.Pipelines.Bbh.FindHorizon import find_horizon, find_horizon_command
from spectre.PointwiseFunctions.AnalyticSolutions.GeneralRelativity import (
    KerrSchild,
)
from spectre.PointwiseFunctions.GeneralRelativity import ricci_tensor
from spectre.Spectral import Basis, Mesh, Quadrature, logical_coordinates
from spectre.SphericalHarmonics import Strahlkorper, cartesian_coords
from spectre.support.Logging import configure_logging


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


def _compute_derived(tensors, mesh, inv_jacobian):
    """Compute the spatial Ricci tensor from the spatial Christoffel symbols"""
    christoffels = tensors["SpatialChristoffelSecondKind"]
    deriv_christoffels = partial_derivative(christoffels, mesh, inv_jacobian)
    tensors["SpatialRicci"] = ricci_tensor(christoffels, deriv_christoffels)
    return tensors


class TestFindHorizon(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(
            unit_test_build_path(), "Pipelines/Bbh/FindHorizon"
        )
        self.h5_filename = os.path.join(self.test_dir, "TestData.h5")
        self.output_surfaces_filename = os.path.join(
            self.test_dir, "Surfaces.h5"
        )
        self.output_reductions_filename = os.path.join(
            self.test_dir, "Reductions.h5"
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
        tensor_names = [
            "SpatialMetric",
            "InverseSpatialMetric",
            "ExtrinsicCurvature",
            "SpatialChristoffelSecondKind",
            # "SpatialRicci", < have to compute this from numeric derivative
        ]
        with spectre_h5.H5File(self.h5_filename, "w") as h5file:
            volfile = h5file.insert_vol("element_data", version=0)
            volfile.write_volume_data(
                observation_id=0,
                observation_value=0.0,
                elements=[
                    spectre_h5.ElementVolumeData(
                        element.id,
                        _to_tensor_components(
                            _compute_derived(
                                solution.variables(
                                    element.inertial_coordinates,
                                    tensor_names,
                                ),
                                element.mesh,
                                element.inv_jacobian,
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

    def test_find_horizon(self):
        horizon, quantities = find_horizon(
            self.h5_filename,
            subfile_name="element_data",
            obs_id=0,
            obs_time=0.0,
            initial_guess=Strahlkorper(
                l_max=12, radius=2.5, center=[0.0, 0.0, 0.0]
            ),
        )
        # Horizon should be a sphere of coordinate radius 2.0
        npt.assert_allclose(
            np.linalg.norm(cartesian_coords(horizon), axis=0), 2.0, atol=1e-3
        )
        # Mass and spin should be 1.0 and 0.0
        npt.assert_allclose(quantities["ChristodoulouMass"], 1.0, atol=1e-3)
        npt.assert_allclose(
            quantities["DimensionlessSpinMagnitude"], 0.0, atol=1e-3
        )

    def test_cli(self):
        runner = CliRunner()
        result = runner.invoke(
            find_horizon_command,
            [
                self.h5_filename,
                "-d",
                "element_data",
                "--step",
                "0",
                "--l-max",
                "12",
                "--initial-radius",
                "2.5",
                "--center",
                "0.0",
                "0.0",
                "0.0",
                "--output-surfaces-file",
                self.output_surfaces_filename,
                "--output-coeffs-subfile",
                "HorizonCoeffs",
                "--output-coords-subfile",
                "HorizonCoords",
                "--output-reductions-file",
                self.output_reductions_filename,
                "--output-quantities-subfile",
                "HorizonQuantities",
            ],
            catch_exceptions=True,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(os.path.exists(self.output_surfaces_filename))
        self.assertTrue(os.path.exists(self.output_reductions_filename))


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
