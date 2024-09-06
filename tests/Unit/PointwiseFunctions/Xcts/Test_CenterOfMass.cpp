// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Slice.hpp"
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

/*
  The tests below shift the isotropic Schwarzschild solution and check that the
  center of mass corresponds to the coordinate shift.

  We have found that the center of mass integral often diverges from the
  expected result as we increase the outer radius. This could be due to
  round-off errors in the calculation, making the result less accurate. Here,
  we're simply checking that this deviation is below some fixed tolerance.
*/

constexpr double TOLERANCE = 1.e-3;

void test_infinite_surface_integral(const double distance,
                                    const double z_shift) {
  // Get Schwarzschild solution in isotropic coordinates.
  const double mass = 1;
  const Xcts::Solutions::Schwarzschild solution(
      mass, Xcts::Solutions::SchwarzschildCoordinates::Isotropic);
  const double horizon_radius = 0.5 * mass;

  // Set up domain
  const size_t h_refinement = 1;
  const size_t p_refinement = 6;
  const domain::creators::Sphere shell{
      /* inner_radius */ z_shift + 1.1 * horizon_radius,
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
          tmpl::list<Xcts::Tags::ConformalFactor<DataVector>>{});
      const auto& conformal_factor =
          get<Xcts::Tags::ConformalFactor<DataVector>>(shifted_fields);

      // Compute area element
      const auto flat_area_element =
          euclidean_area_element(inv_jacobian, boundary_direction);

      // Evaluate surface integral
      const auto surface_integrand = Xcts::center_of_mass_surface_integrand(
          conformal_factor, inertial_coords);
      for (int I = 0; I < 3; I++) {
        surface_integral.get(I) += definite_integral(
            surface_integrand.get(I) * get(flat_area_element), face_mesh);
      }
    }
  }

  // Check result
  auto custom_approx = Approx::custom().epsilon(TOLERANCE).scale(1.0);
  CHECK(get<0>(surface_integral) == custom_approx(0.));
  CHECK(get<1>(surface_integral) == custom_approx(0.));
  CHECK(get<2>(surface_integral) / mass == custom_approx(z_shift));
}

void test_infinite_volume_integral(const double distance,
                                   const double z_shift) {
  // Get Schwarzschild solution in isotropic coordinates.
  const double mass = 1;
  const Xcts::Solutions::Schwarzschild solution(
      mass, Xcts::Solutions::SchwarzschildCoordinates::Isotropic);
  const double horizon_radius = 0.5 * mass;

  // Set up domain
  const size_t h_refinement = 1;
  const size_t p_refinement = 11;
  const domain::creators::Sphere shell{
      /* inner_radius */ z_shift + 1.1 * horizon_radius,
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
  const Mesh<3> mesh{p_refinement + 1, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const Mesh<2> face_mesh{p_refinement + 1, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};

  // Initialize "reduced" integral
  tnsr::I<double, 3> total_integral({0., 0., 0.});

  // Compute integrals by summing over each element
  for (const auto& element_id : element_ids) {
    // Get element information
    const auto& current_block = blocks.at(element_id.block_id());
    const auto current_element = domain::Initialization::create_initial_element(
        element_id, current_block, initial_ref_levels);
    const ElementMap<3, Frame::Inertial> logical_to_inertial_map(
        element_id, current_block.stationary_map().get_clone());

    // Get 3D coordinates
    const auto logical_coords = logical_coordinates(mesh);
    const auto inertial_coords = logical_to_inertial_map(logical_coords);
    const auto jacobian = logical_to_inertial_map.jacobian(logical_coords);
    const auto det_jacobian = determinant(jacobian);

    // Shift coordinates used to get analytic solution
    auto shifted_coords = inertial_coords;
    shifted_coords.get(2) -= z_shift;

    // Get required fields
    const auto shifted_fields = solution.variables(
        shifted_coords,
        tmpl::list<
            Xcts::Tags::ConformalFactor<DataVector>,
            ::Tags::deriv<Xcts::Tags::ConformalFactorMinusOne<DataVector>,
                          tmpl::size_t<3>, Frame::Inertial>>{});
    const auto& conformal_factor =
        get<Xcts::Tags::ConformalFactor<DataVector>>(shifted_fields);
    const auto& deriv_conformal_factor =
        get<::Tags::deriv<Xcts::Tags::ConformalFactorMinusOne<DataVector>,
                          tmpl::size_t<3>, Frame::Inertial>>(shifted_fields);

    // Evaluate volume integral.
    const auto volume_integrand = Xcts::center_of_mass_volume_integrand(
        conformal_factor, deriv_conformal_factor, inertial_coords);
    for (int I = 0; I < 3; I++) {
      total_integral.get(I) +=
          definite_integral(volume_integrand.get(I) * get(det_jacobian), mesh);
    }

    // Loop over external boundaries
    for (auto boundary_direction : current_element.external_boundaries()) {
      // Skip interfaces not at the inner boundary
      if (boundary_direction != Direction<3>::lower_zeta()) {
        continue;
      }

      // Get interface coordinates.
      const auto face_logical_coords =
          interface_logical_coordinates(face_mesh, boundary_direction);
      const auto face_inv_jacobian =
          logical_to_inertial_map.inv_jacobian(face_logical_coords);

      // Slice required fields to the interface
      const size_t slice_index =
          index_to_slice_at(mesh.extents(), boundary_direction);
      const auto& face_conformal_factor =
          data_on_slice(conformal_factor, mesh.extents(),
                        boundary_direction.dimension(), slice_index);
      const auto& face_inertial_coords =
          data_on_slice(inertial_coords, mesh.extents(),
                        boundary_direction.dimension(), slice_index);

      // Compute Euclidean area element
      const auto flat_area_element =
          euclidean_area_element(face_inv_jacobian, boundary_direction);

      // Evaluate surface integral.
      const auto surface_integrand = Xcts::center_of_mass_surface_integrand(
          face_conformal_factor, face_inertial_coords);
      for (int I = 0; I < 3; I++) {
        total_integral.get(I) += definite_integral(
            surface_integrand.get(I) * get(flat_area_element), face_mesh);
      }
    }
  }

  // Check result
  auto custom_approx = Approx::custom().epsilon(TOLERANCE).scale(1.0);
  CHECK(get<0>(total_integral) == custom_approx(0.));
  CHECK(get<1>(total_integral) == custom_approx(0.));
  CHECK(get<2>(total_integral) / mass == custom_approx(z_shift));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Xcts.CenterOfMass",
                  "[Unit][PointwiseFunctions]") {
  for (const double distance : std::array<double, 3>({1.e3, 1.e4, 1.e5})) {
    test_infinite_surface_integral(distance, 0.);
    test_infinite_surface_integral(distance, 0.1);
    test_infinite_volume_integral(distance, 0.);
    test_infinite_volume_integral(distance, 0.1);
  }
}
