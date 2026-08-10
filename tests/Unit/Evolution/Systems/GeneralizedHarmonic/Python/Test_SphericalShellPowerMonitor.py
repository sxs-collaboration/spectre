# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

import numpy as np
import numpy.testing as npt

import spectre.PointwiseFunctions.GeneralRelativity as gr
import spectre.PointwiseFunctions.GeneralRelativity.GeneralizedHarmonic as gh
from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import Frame, tnsr
from spectre.Domain import ElementId
from spectre.Domain.Creators import BinaryCompactObject, SphericalShells
from spectre.Domain.Creators.TimeDependentOptions import (
    BinaryCompactObjectTimeDependentOptions,
    ExpansionMapOptions,
)
from spectre.Evolution.Systems.GeneralizedHarmonic import (
    gh_shell_power_monitors,
)
from spectre.PointwiseFunctions.AnalyticSolutions.GeneralRelativity import (
    KerrSchild,
)
from spectre.Spectral import Basis, Mesh, Quadrature, logical_coordinates


class TestSphericalShellPowerMonitor(unittest.TestCase):
    @staticmethod
    def binary_compact_object(time_dependent_options):
        return BinaryCompactObject(
            inner_radius_a=0.5,
            outer_radius_a=2.0,
            x_coord_a=5.0,
            excise_a=True,
            use_logarithmic_map_a=True,
            inner_radius_b=0.5,
            outer_radius_b=2.0,
            x_coord_b=-5.0,
            excise_b=True,
            use_logarithmic_map_b=True,
            center_of_mass_offset=[0.0, 0.0],
            envelope_radius=50.0,
            outer_radius=100.0,
            cube_scale=1.2,
            initial_refinement=0,
            initial_number_of_grid_points=8,
            use_equiangular_map=True,
            radial_partitioning_outer_shell=[75.0],
            opening_angle_in_degrees=120.0,
            spherical_harmonics_in_wavezone=True,
            use_worldtube=False,
            time_dependent_options=time_dependent_options,
        )

    def test_gh_shell_power_monitors(self):
        inner_radius = 4.0
        outer_radius = 7.0
        radial_points = 7
        l_max = 8
        mesh = Mesh[3](
            [radial_points, l_max + 1, 2 * l_max + 1],
            [
                Basis.Legendre,
                Basis.SphericalHarmonic,
                Basis.SphericalHarmonic,
            ],
            [
                Quadrature.GaussLobatto,
                Quadrature.Gauss,
                Quadrature.Equiangular,
            ],
        )
        logical_coords = np.asarray(logical_coordinates(mesh))
        radius = 0.5 * (outer_radius - inner_radius) * logical_coords[
            0
        ] + 0.5 * (outer_radius + inner_radius)
        theta = logical_coords[1]
        azimuthal_angle = logical_coords[2]
        cartesian_coords = tnsr.I[DataVector, 3, Frame.Inertial](
            np.array(
                [
                    radius * np.sin(theta) * np.cos(azimuthal_angle),
                    radius * np.sin(theta) * np.sin(azimuthal_angle),
                    radius * np.cos(theta),
                ]
            )
        )

        solution = KerrSchild(
            mass=1.0,
            dimensionless_spin=[0.05, -0.1, 0.15],
            center=[0.0, 0.0, 0.0],
        )
        solution_vars = solution.variables(
            cartesian_coords,
            [
                "Lapse",
                "dt(Lapse)",
                "deriv(Lapse)",
                "Shift",
                "dt(Shift)",
                "deriv(Shift)",
                "SpatialMetric",
                "dt(SpatialMetric)",
                "deriv(SpatialMetric)",
            ],
        )

        spacetime_metric = gr.spacetime_metric(
            solution_vars["Lapse"],
            solution_vars["Shift"],
            solution_vars["SpatialMetric"],
        )
        phi_tensor = gh.phi(
            solution_vars["Lapse"],
            solution_vars["deriv(Lapse)"],
            solution_vars["Shift"],
            solution_vars["deriv(Shift)"],
            solution_vars["SpatialMetric"],
            solution_vars["deriv(SpatialMetric)"],
        )
        pi = gh.pi(
            solution_vars["Lapse"],
            solution_vars["dt(Lapse)"],
            solution_vars["Shift"],
            solution_vars["dt(Shift)"],
            solution_vars["SpatialMetric"],
            solution_vars["dt(SpatialMetric)"],
            phi_tensor,
        )

        domain = SphericalShells(
            inner_radius, outer_radius, 0, radial_points, l_max
        ).create_domain()

        monitors = gh_shell_power_monitors(
            spacetime_metric,
            pi,
            phi_tensor,
            mesh,
            ElementId[3](0),
            domain,
            0.0,
            {},
        )

        incompatible_mesh = Mesh[3](
            mesh.extents(),
            [Basis.Legendre, Basis.Legendre, Basis.Legendre],
            [
                Quadrature.GaussLobatto,
                Quadrature.GaussLobatto,
                Quadrature.GaussLobatto,
            ],
        )
        with self.assertRaisesRegex(
            RuntimeError, "require the mesh dimensions"
        ):
            gh_shell_power_monitors(
                spacetime_metric,
                pi,
                phi_tensor,
                incompatible_mesh,
                ElementId[3](0),
                domain,
                0.0,
                {},
            )

        # These hard-coded power monitors are from SpEC, as described in
        # SphericalShellPowerMonitor.cpp.
        expected_monitors = {
            "SpacetimeMetric": {
                "radial": [
                    0.66272576586915166,
                    0.055118989877186962,
                    0.010223513029798308,
                    0.001706566677470182,
                    0.00027144507714939298,
                    0.000042905006422366406,
                    0.0000063892936284495129,
                ],
                "angular": [
                    1.9854935257947095,
                    0.00806505390676714,
                    0.00016837816034157,
                    0.00000418589096616,
                    0.0000001661210045,
                    0.00000000552502198,
                    0.00000000024433662,
                    0.00000000000788883,
                    0.00000000000025781,
                ],
            },
            "Pi": {
                "radial": [
                    0.012573571910520885,
                    0.009832206078082233,
                    0.0034601774215344341,
                    0.00091860759247559592,
                    0.00021004678157227581,
                    0.000045199806501584563,
                    0.0000085382443326196344,
                ],
                "angular": [
                    0.055211802785050945,
                    0.00068180518923684,
                    0.00004171340786288,
                    0.00000087524448329,
                    0.00000007098264593,
                    0.00000000191450705,
                    0.00000000014586247,
                    0.00000000000382953,
                    0.00000000000019612,
                ],
            },
            "Phi": {
                "radial": [
                    0.03582086057745523,
                    0.01984177151721315,
                    0.00550825042019753,
                    0.00122441336098692,
                    0.00024331209932814,
                    0.00004644598374878,
                    0.00000801247933444,
                ],
                "angular": [
                    0.15154167256606327,
                    0.00335975215390938,
                    0.00010001010476677,
                    0.00000325894876587,
                    0.00000016423770303,
                    0.00000000657476944,
                    0.00000000029571569,
                    0.0000000000077388,
                    0.00000000000002565,
                ],
            },
        }
        self.assertEqual(set(monitors.keys()), set(expected_monitors.keys()))
        for variable_name, expected_monitor in expected_monitors.items():
            self.assertEqual(
                set(monitors[variable_name].keys()), {"radial", "angular"}
            )
            for dimension, expected_power in expected_monitor.items():
                npt.assert_allclose(
                    np.asarray(monitors[variable_name][dimension]),
                    expected_power,
                    rtol=1.0e-11,
                    atol=1.0e-15,
                )

    # Compare power monitors computed with a moving domain to those computed
    # with a stationary domain after manually transforming the tensors.
    def test_time_dependent_grid_to_inertial_jacobian(self):
        mesh = Mesh[3](
            [3, 4, 7],
            [
                Basis.Legendre,
                Basis.SphericalHarmonic,
                Basis.SphericalHarmonic,
            ],
            [
                Quadrature.GaussLobatto,
                Quadrature.Gauss,
                Quadrature.Equiangular,
            ],
        )
        num_points = mesh.number_of_grid_points()
        logical_coords = logical_coordinates(mesh)
        profile = 1.0 + 0.2 * np.asarray(logical_coords)[0]

        metric_type = tnsr.aa[DataVector, 3, Frame.Inertial]
        phi_type = tnsr.iaa[DataVector, 3, Frame.Inertial]
        spacetime_metric = metric_type(num_points=num_points, fill=0.0)
        pi = metric_type(num_points=num_points, fill=0.0)
        phi = phi_type(num_points=num_points, fill=0.0)
        spacetime_metric[spacetime_metric.get_storage_index(1, 1)] = DataVector(
            profile
        )
        pi[pi.get_storage_index(1, 1)] = DataVector(2.0 * profile)
        phi[phi.get_storage_index(0, 1, 1)] = DataVector(3.0 * profile)

        expansion_map = ExpansionMapOptions([2.0, 0.0, 0.0], 100.0, 0.0)
        time_dependent_options = BinaryCompactObjectTimeDependentOptions(
            0.0,
            expansion_map,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        moving_creator = self.binary_compact_object(time_dependent_options)
        moving_domain = moving_creator.create_domain()
        functions_of_time = moving_creator.functions_of_time()
        shell_block_id = 34
        shell_element_id = ElementId[3](shell_block_id)

        block_logical_coords = tnsr.I[DataVector, 3, Frame.BlockLogical](
            np.asarray(logical_coords)
        )
        shell_block = moving_domain.blocks[shell_block_id]
        grid_coords = shell_block.moving_mesh_logical_to_grid_map(
            block_logical_coords
        )
        jacobian = shell_block.moving_mesh_grid_to_inertial_map.jacobian(
            grid_coords, 0.0, functions_of_time
        )
        jacobian_data = np.array(
            [
                [np.asarray(jacobian.get(i, j)) for j in range(3)]
                for i in range(3)
            ]
        )
        identity = np.identity(3)[:, :, None]
        self.assertGreater(np.max(np.abs(jacobian_data - identity)), 0.1)

        transformed_metric = metric_type(num_points=num_points, fill=0.0)
        transformed_pi = metric_type(num_points=num_points, fill=0.0)
        transformed_phi = phi_type(num_points=num_points, fill=0.0)
        for j in range(3):
            for k in range(j, 3):
                transformed_component = (
                    jacobian_data[0, j] * jacobian_data[0, k] * profile
                )
                transformed_metric[
                    transformed_metric.get_storage_index(j + 1, k + 1)
                ] = DataVector(transformed_component)
                transformed_pi[
                    transformed_pi.get_storage_index(j + 1, k + 1)
                ] = DataVector(2.0 * transformed_component)
            for k in range(3):
                for ell in range(k, 3):
                    transformed_phi[
                        transformed_phi.get_storage_index(j, k + 1, ell + 1)
                    ] = DataVector(
                        3.0
                        * jacobian_data[0, j]
                        * jacobian_data[0, k]
                        * jacobian_data[0, ell]
                        * profile
                    )

        moving_monitors = gh_shell_power_monitors(
            spacetime_metric,
            pi,
            phi,
            mesh,
            shell_element_id,
            moving_domain,
            0.0,
            functions_of_time,
        )
        stationary_monitors = gh_shell_power_monitors(
            transformed_metric,
            transformed_pi,
            transformed_phi,
            mesh,
            shell_element_id,
            self.binary_compact_object(None).create_domain(),
            0.0,
            {},
        )
        for variable_name in ("SpacetimeMetric", "Pi", "Phi"):
            npt.assert_allclose(
                moving_monitors[variable_name]["angular"],
                stationary_monitors[variable_name]["angular"],
                rtol=2.0e-13,
                atol=2.0e-14,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
