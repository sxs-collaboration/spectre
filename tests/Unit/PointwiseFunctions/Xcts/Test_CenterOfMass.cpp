// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/AreaElement.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/ElementToBlockLogicalMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Schwarzschild.hpp"
#include "PointwiseFunctions/Xcts/CenterOfMass.hpp"

namespace {

/**
 * This test shifts the isotropic Schwarzschild solution and checks that the
 * center of mass corresponds to the coordinate shift.
 */
void test_center_of_mass_surface_integral(const double distance) {
  // Get Schwarzschild solution in isotropic coordinates.
  const double mass = 1;
  const Xcts::Solutions::Schwarzschild solution(
      mass, Xcts::Solutions::SchwarzschildCoordinates::Isotropic);
  const double horizon_radius = 0.5 * mass;

  // Define z-shift applied to the coordinates.
  const double z_shift = 2. * mass;

  // Set up domain
  const size_t h_refinement = 1;
  const size_t p_refinement = 6;
  const domain::creators::Sphere shell{
      /* inner_radius */ z_shift + horizon_radius,
      /* outer_radius */ distance,
      /* interior */ domain::creators::Sphere::Excision{},
      /* initial_refinement */ h_refinement,
      /* initial_number_of_grid_points */ p_refinement + 1,
      /* use_equiangular_map */ true,
      /* equatorial_compression */ {},
      /* radial_partitioning */ {},
      /* radial_distribution */ domain::CoordinateMaps::Distribution::Inverse};
  const auto shell_domain = shell.create_domain();
  const auto& blocks = shell_domain.blocks();
  const auto& initial_ref_levels = shell.initial_refinement_levels();
  const auto element_ids = initial_element_ids(initial_ref_levels);
  const Mesh<2> face_mesh{p_refinement + 1, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};

  // Initialize surface integral
  tnsr::I<double, 3> surface_integral({0., 0., 0.});

  // Compute integrals by summing over each element
  for (const auto& element_id : element_ids) {
    // Skip elements not at the outer boundary
    const size_t radial_dimension = 2;
    const auto radial_segment_id = element_id.segment_id(radial_dimension);
    if (radial_segment_id.index() !=
        two_to_the(radial_segment_id.refinement_level()) - 1) {
      continue;
    }

    // Get element information
    const auto& current_block = blocks.at(element_id.block_id());
    const auto current_element = domain::Initialization::create_initial_element(
        element_id, current_block, initial_ref_levels);
    const ElementMap<3, Frame::Inertial> logical_to_inertial_map(
        element_id, current_block.stationary_map().get_clone());

    // Loop over external boundaries
    for (auto boundary_direction : current_element.external_boundaries()) {
      // Skip interfaces not at the outer boundary
      if (boundary_direction != Direction<3>::upper_zeta()) {
        continue;
      }

      // Get interface coordinates
      const auto logical_coords =
          interface_logical_coordinates(face_mesh, boundary_direction);
      const auto inertial_coords = logical_to_inertial_map(logical_coords);
      const auto inv_jacobian =
          logical_to_inertial_map.inv_jacobian(logical_coords);

      // Shift coordinates used to get analytic solution
      auto shifted_coords = inertial_coords;
      shifted_coords.get(2) -= z_shift;

      // Get required fields on the interface
      const auto shifted_fields = solution.variables(
          shifted_coords,
          tmpl::list<
              Xcts::Tags::ConformalFactor<DataVector>,
              Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
              Xcts::Tags::InverseConformalMetric<DataVector, 3,
                                                 Frame::Inertial>>{});
      const auto& conformal_factor =
          get<Xcts::Tags::ConformalFactor<DataVector>>(shifted_fields);
      const auto& conformal_metric =
          get<Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>>(
              shifted_fields);
      const auto& inv_conformal_metric = get<
          Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>>(
          shifted_fields);

      // Compute outward-pointing unit normal.
      const auto conformal_r = magnitude(inertial_coords, conformal_metric);
      const auto conformal_unit_normal =
          tenex::evaluate<ti::I>(inertial_coords(ti::I) / conformal_r());

      // Compute area element
      const auto sqrt_det_conformal_metric =
          Scalar<DataVector>(sqrt(get(determinant(conformal_metric))));
      const auto conformal_area_element =
          area_element(inv_jacobian, boundary_direction, inv_conformal_metric,
                       sqrt_det_conformal_metric);

      // Integrate
      const auto surface_integrand = Xcts::center_of_mass_surface_integrand(
          conformal_factor, conformal_unit_normal);
      for (int I = 0; I < 3; I++) {
        surface_integral.get(I) += definite_integral(
            surface_integrand.get(I) * get(conformal_area_element), face_mesh);
      }
    }
  }

  // Check result
  auto custom_approx = Approx::custom().epsilon(10. / distance).scale(1.0);
  CHECK(get<0>(surface_integral) == custom_approx(0.));
  CHECK(get<1>(surface_integral) == custom_approx(0.));
  CHECK(get<2>(surface_integral) / mass == custom_approx(z_shift));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Xcts.CenterOfMass",
                  "[Unit][PointwiseFunctions]") {
  // Test that integral converges with distance.
  for (const double distance : std::array<double, 3>({1.e3, 1.e4, 1.e5})) {
    test_center_of_mass_surface_integral(distance);
  }
}
